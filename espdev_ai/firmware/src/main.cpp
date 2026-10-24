#include <Arduino.h>
#include <FS.h>
#include <SPIFFS.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_spi_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "tokenizer_runtime.h"
#include "model_offsets.h"
#include "model_layout.h"
#include "mbedtls/sha256.h"

                                                                                               
                                                                                                  
                                                                                              
namespace cfg = pocketgpt_final_v7b50;
namespace tied = pocketgpt_tied30m;

namespace {

constexpr float RMS_EPS = 1.0e-5f;
constexpr float ROPE_THETA = 10000.0f;
constexpr std::size_t TOP_K = 1;
constexpr std::size_t MAX_PROMPT_TOKENS = 64;
constexpr std::size_t MAX_DECODE_BYTES = 512;
constexpr std::size_t MAX_GENERATE_TOKENS = 24;

constexpr std::uint32_t FINAL_USER_ID = 4u;
constexpr std::uint32_t FINAL_ASSISTANT_ID = 5u;
constexpr std::uint32_t FINAL_END_ID = 9u;

                                         
                                              
                     
constexpr bool FINAL_STOP_AT_FIRST_SENTENCE = false;


struct MatrixBuf {
  const char* label;
  std::uint8_t* q4;
  std::uint16_t* scales;
  std::uint32_t q4_capacity;
  std::uint32_t scales_capacity;
  cfg::MatrixSpec spec;
};

alignas(16) float hidden[cfg::kDModel];
alignas(16) float normed[cfg::kDModel];
alignas(16) float q_vec[cfg::kDModel];
alignas(16) float k_vec[cfg::kKvDim];
alignas(16) float v_vec[cfg::kKvDim];
alignas(16) float attention[cfg::kDModel];
alignas(16) float o_vec[cfg::kDModel];
alignas(16) float gate_vec[cfg::kDff];
alignas(16) float up_vec[cfg::kDff];
alignas(16) float down_vec[cfg::kDModel];
alignas(16) float attn_norm_weight[cfg::kDModel];
alignas(16) float ffn_norm_weight[cfg::kDModel];
alignas(16) float final_norm_weight[cfg::kDModel];
alignas(16) float final_normed[cfg::kDModel];
alignas(16) float attn_scores[MAX_PROMPT_TOKENS];
alignas(16) float rope_cos_cache[cfg::kHeadDim / 2];
alignas(16) float rope_sin_cache[cfg::kHeadDim / 2];

std::size_t g_rope_cached_position =
    static_cast<std::size_t>(-1);

std::uint64_t g_rope_cache_build_us = 0;
std::uint64_t g_rope_cache_hits = 0;
std::uint64_t g_rope_cache_misses = 0;
alignas(16) float attn_scores_worker[MAX_PROMPT_TOKENS];
static std::uint16_t fp16_row[cfg::kDModel];

MatrixBuf mb_q;
MatrixBuf mb_k;
MatrixBuf mb_v;
MatrixBuf mb_o;
MatrixBuf mb_gate;
MatrixBuf mb_up;
MatrixBuf mb_down;

const esp_partition_t* part_q4 = nullptr;
const esp_partition_t* part_scales = nullptr;
const esp_partition_t* part_kept = nullptr;

float* lm_head_cache_fp32 = nullptr;
std::size_t lm_head_cache_bytes = 0;

                                                    
                                                                    
float* q4_scales_cache_fp32 = nullptr;
std::size_t q4_scales_cache_count = 0;

std::uint64_t g_layer_load_us = 0;
std::uint64_t g_layer_map_us = 0;
std::uint32_t g_layer_mmap_fallbacks = 0;

std::uint64_t g_attn_norm_us = 0;
std::uint64_t g_q_us = 0;
std::uint64_t g_q_pair_us = 0;
std::uint64_t g_q_main_work_us = 0;
std::uint64_t g_q_worker_work_us = 0;
std::uint64_t g_k_us = 0;
std::uint64_t g_v_us = 0;
std::uint64_t g_kv_pair_us = 0;
std::uint64_t g_attn_us = 0;
std::uint64_t g_rope_us = 0;
std::uint64_t g_kvstore_us = 0;
std::uint64_t g_attn_pair_us = 0;
std::uint64_t g_attn_main_work_us = 0;
std::uint64_t g_attn_worker_work_us = 0;
std::uint64_t g_o_us = 0;
std::uint64_t g_o_pair_us = 0;
std::uint64_t g_o_main_work_us = 0;
std::uint64_t g_o_worker_work_us = 0;
std::uint64_t g_ffn_norm_us = 0;
std::uint64_t g_gate_us = 0;
std::uint64_t g_up_us = 0;

std::uint64_t g_gateup_pair_us = 0;

struct GateUpWorkerJob {
  bool use_mmap;
  bool row_range;

  bool attention_job;
  std::size_t attention_layer_id;
  std::size_t attention_position;
  std::size_t row_begin;
  std::size_t row_end;
  const cfg::MatrixSpec* spec;
  const MatrixBuf* fallback;
  const std::uint8_t* q_ptr;
  const std::uint16_t* scales_ptr;
  const float* input;
  float* output;
  volatile std::uint32_t elapsed_us;
};

TaskHandle_t g_gateup_worker_task = nullptr;
SemaphoreHandle_t g_gateup_done = nullptr;
GateUpWorkerJob g_gateup_job{};
std::uint64_t g_swiglu_us = 0;
std::uint64_t g_swiglu_pair_us = 0;
std::uint64_t g_swiglu_main_work_us = 0;
std::uint64_t g_swiglu_worker_work_us = 0;
std::uint64_t g_down_us = 0;
std::uint64_t g_down_pair_us = 0;
std::uint64_t g_down_main_work_us = 0;
std::uint64_t g_down_worker_work_us = 0;

float* kv_cache_k = nullptr;
float* kv_cache_v = nullptr;

pocketlm::TokenizerRuntime tokenizer{};

static inline float fp16_to_float(std::uint16_t bits) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(bits >> 15);

  const std::uint32_t exponent =
      static_cast<std::uint32_t>((bits >> 10) & 0x1Fu);

  const std::uint32_t mantissa =
      static_cast<std::uint32_t>(bits & 0x03FFu);

  std::uint32_t fp32_bits = 0;

  if (exponent == 0) {
    if (mantissa == 0) {
      fp32_bits = sign << 31;
    } else {
      std::uint32_t mant = mantissa;
      int exp = -14;

      while ((mant & 0x0400u) == 0) {
        mant <<= 1;
        --exp;
      }

      mant &= 0x03FFu;

      const std::uint32_t fp32_exp =
          static_cast<std::uint32_t>(exp + 127);

      fp32_bits =
          (sign << 31)
          | (fp32_exp << 23)
          | (mant << 13);
    }
  } else if (exponent == 0x1Fu) {
    fp32_bits =
        (sign << 31)
        | (0xFFu << 23)
        | (mantissa << 13);
  } else {
    const std::uint32_t fp32_exp =
        exponent + (127 - 15);

    fp32_bits =
        (sign << 31)
        | (fp32_exp << 23)
        | (mantissa << 13);
  }

  float result;
  std::memcpy(&result, &fp32_bits, sizeof(result));
  return result;
}

std::int8_t decode_i4(std::uint8_t nibble) {
  nibble &= 0x0Fu;
  return (nibble & 0x08u) ? static_cast<std::int8_t>(nibble) - 16 : static_cast<std::int8_t>(nibble);
}

void init_matrix_buf(MatrixBuf* mb, const char* label) {
  mb->label = label;
  mb->q4 = nullptr;
  mb->scales = nullptr;
  mb->q4_capacity = 0;
  mb->scales_capacity = 0;
  std::memset(&mb->spec, 0, sizeof(mb->spec));
}

void init_matrix_bufs() {
  init_matrix_buf(&mb_q, "q_proj");
  init_matrix_buf(&mb_k, "k_proj");
  init_matrix_buf(&mb_v, "v_proj");
  init_matrix_buf(&mb_o, "o_proj");
  init_matrix_buf(&mb_gate, "gate_proj");
  init_matrix_buf(&mb_up, "up_proj");
  init_matrix_buf(&mb_down, "down_proj");
}

void* alloc_bytes(std::size_t bytes, const char* label) {
  void* p = heap_caps_malloc(bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (!p) p = heap_caps_malloc(bytes, MALLOC_CAP_8BIT);
  if (!p) Serial.printf("[FAIL] allocation failed for %s bytes=%lu\n", label, static_cast<unsigned long>(bytes));
  (void)label;
  return p;
}

bool ensure_matrix_capacity(MatrixBuf& mb, const cfg::MatrixSpec& spec) {
  if (!mb.q4 || mb.q4_capacity < spec.q4_bytes) {
    if (mb.q4) heap_caps_free(mb.q4);
    mb.q4 = static_cast<std::uint8_t*>(alloc_bytes(spec.q4_bytes, mb.label));
    mb.q4_capacity = spec.q4_bytes;
  }
  if (!mb.scales || mb.scales_capacity < spec.scales_bytes) {
    if (mb.scales) heap_caps_free(mb.scales);
    char label[40];
    snprintf(label, sizeof(label), "%s.scales", mb.label);
    mb.scales = static_cast<std::uint16_t*>(alloc_bytes(spec.scales_bytes, label));
    mb.scales_capacity = spec.scales_bytes;
  }
  return mb.q4 && mb.scales;
}

bool read_exact(const esp_partition_t* part, std::size_t offset, void* dst, std::size_t bytes, const char* label) {
  if (!part || !dst) return false;
  if ((uint64_t)offset + (uint64_t)bytes > (uint64_t)part->size) {
    Serial.printf("[FAIL] read out of range %s offset=%lu bytes=%lu part_size=%lu\n", label, static_cast<unsigned long>(offset), static_cast<unsigned long>(bytes), static_cast<unsigned long>(part->size));
    return false;
  }
  const esp_err_t err = esp_partition_read(part, offset, dst, bytes);
  if (err != ESP_OK) {
    Serial.printf("[FAIL] esp_partition_read %s offset=%lu bytes=%lu err=%d\n", label, static_cast<unsigned long>(offset), static_cast<unsigned long>(bytes), static_cast<int>(err));
    return false;
  }
  return true;
}

const esp_partition_t* find_part(const char* label, std::size_t expected_min) {
  const esp_partition_t* part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label);
  if (!part) { Serial.printf("[FAIL] partition not found: %s\n", label); return nullptr; }
  if (part->size < expected_min) { Serial.printf("[FAIL] partition too small: %s\n", label); return nullptr; }
  return part;
}

bool read_fp16_vector(const esp_partition_t* part, std::size_t offset, float* out, std::size_t count, const char* label) {
  if (!part || !out) return false;
  if (count > cfg::kDModel) { Serial.printf("[FAIL] fp16 temp too small for %s count=%lu\n", label, static_cast<unsigned long>(count)); return false; }
  if (!read_exact(part, offset, fp16_row, count * sizeof(std::uint16_t), label)) return false;
  for (std::size_t i = 0; i < count; ++i) out[i] = fp16_to_float(fp16_row[i]);
  return true;
}

bool read_embedding_row(std::uint32_t token_id, float* out) {
  if (token_id >= cfg::kVocabSize) return false;
  const std::size_t offset = cfg::kTokEmbKeptOffset + static_cast<std::size_t>(token_id) * cfg::kTokEmbeddingRowBytes;
  return read_fp16_vector(part_kept, offset, out, cfg::kDModel, "tok_embedding");
}


bool cache_lm_head_to_psram() {
  const std::size_t elements =
      static_cast<std::size_t>(cfg::kVocabSize)
      * static_cast<std::size_t>(cfg::kDModel);

  lm_head_cache_bytes =
      elements * sizeof(float);

  lm_head_cache_fp32 =
      static_cast<float*>(
          heap_caps_malloc(
              lm_head_cache_bytes,
              MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));

  if (!lm_head_cache_fp32) {
    Serial.printf(
        "[LMHEAD_FP32_CACHE] FAIL alloc bytes=%lu free_psram=%lu\n",
        static_cast<unsigned long>(lm_head_cache_bytes),
        static_cast<unsigned long>(ESP.getFreePsram()));
    return false;
  }

  const std::uint32_t start_us = micros();

  for (std::uint32_t tok = 0;
       tok < cfg::kVocabSize;
       ++tok) {

    const std::size_t off =
        tied::kLmHeadKeptOffset
        + static_cast<std::size_t>(tok)
          * tied::kLmHeadRowBytes;

    if (!read_exact(
            part_kept,
            off,
            fp16_row,
            tied::kLmHeadRowBytes,
            "lm_head_fp32_cache")) {
      return false;
    }

    float* dst =
        lm_head_cache_fp32
        + static_cast<std::size_t>(tok)
          * cfg::kDModel;

    for (std::size_t i = 0;
         i < cfg::kDModel;
         ++i) {
      dst[i] = fp16_to_float(fp16_row[i]);
    }

    if ((tok & 0xFFu) == 0) {
      yield();
    }
  }

  const std::uint32_t load_us =
      micros() - start_us;

  const std::uint8_t* bytes =
      reinterpret_cast<const std::uint8_t*>(
          lm_head_cache_fp32);

  std::uint32_t sum = 0;

  const std::size_t sample =
      lm_head_cache_bytes < 4096
          ? lm_head_cache_bytes
          : 4096;

  for (std::size_t i = 0;
       i < sample;
       ++i) {
    sum += bytes[i];
  }

  Serial.printf(
      "[LMHEAD_FP32_CACHE] PASS bytes=%lu "
      "load_ms=%.3f first4096_sum=%lu "
      "free_psram=%lu\n",
      static_cast<unsigned long>(
          lm_head_cache_bytes),
      static_cast<float>(load_us) / 1000.0f,
      static_cast<unsigned long>(sum),
      static_cast<unsigned long>(
          ESP.getFreePsram()));

  return true;
}

bool tied30m_fail(const char* reason) {
  Serial.printf("[TIED30M_PACKAGE] FAIL %s\n", reason);
  return false;
}

void print_sha_prefix(const char* label, const std::uint8_t* sha) {
  Serial.printf(" %s=%02x%02x%02x%02x%02x%02x%02x%02x", label, sha[0], sha[1], sha[2], sha[3], sha[4], sha[5], sha[6], sha[7]);
}

bool sha256_partition_prefix(const esp_partition_t* part, std::size_t bytes, std::uint8_t out[32]) {
  constexpr std::size_t kChunk = 16384;
  std::uint8_t* buf = static_cast<std::uint8_t*>(heap_caps_malloc(kChunk, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
  if (!buf) return false;
  mbedtls_sha256_context ctx;
  mbedtls_sha256_init(&ctx);
  bool ok = mbedtls_sha256_starts(&ctx, 0) == 0;
  for (std::size_t off = 0; ok && off < bytes; off += kChunk) {
    const std::size_t n = (bytes - off) < kChunk ? (bytes - off) : kChunk;
    ok = read_exact(part, off, buf, n, "tied30m_sha256") && mbedtls_sha256_update(&ctx, buf, n) == 0;
    if ((off & 0xFFFFFu) == 0) yield();
  }
  ok = ok && mbedtls_sha256_finish(&ctx, out) == 0;
  mbedtls_sha256_free(&ctx);
  heap_caps_free(buf);
  return ok;
}

                                                                                       
                                                                                        
                                                                  
bool verify_tied30m_package() {
  const std::uint32_t start_us = micros();
  std::uint8_t footer[tied::kFooterBytes];
  if (!read_exact(part_kept, tied::kFooterOffset, footer, sizeof(footer), "tied30m_footer")) return tied30m_fail("footer read");
  if (std::memcmp(footer, tied::kFooterMagic, sizeof(tied::kFooterMagic)) != 0) {
    return tied30m_fail("footer magic missing: not a TIED30M dedup package (old 25.75MB or untied 33.21M weights?)");
  }
  std::uint32_t f[6];
  std::memcpy(f, footer + 8, sizeof(f));
  if (f[0] != tied::kLayoutVersion) return tied30m_fail("layout version");
  if (f[1] != tied::kKeptPayloadBytes) return tied30m_fail("kept payload size");
  if (f[2] != tied::kLmHeadKeptOffset) return tied30m_fail("head alias offset");
  if (f[3] != tied::kSharedMatrixBytes) return tied30m_fail("shared matrix size");
  if (f[4] != cfg::kVocabSize || f[5] != cfg::kDModel) return tied30m_fail("shared matrix shape");

  struct Region { const esp_partition_t* part; std::size_t bytes; const std::uint8_t* expected; const char* label; };
  const Region regions[] = {
    {part_q4, cfg::kWeightsQ4Bytes, tied::kWeightsQ4Sha256, "q4"},
    {part_scales, cfg::kScalesF16Bytes, tied::kScalesF16Sha256, "scales"},
    {part_kept, tied::kKeptTensorsBytes, tied::kKeptTensorsSha256, "kept"},
  };
  std::uint8_t sha[3][32];
  for (std::size_t i = 0; i < 3; ++i) {
    if (!sha256_partition_prefix(regions[i].part, regions[i].bytes, sha[i])) return tied30m_fail("sha256 read");
    if (std::memcmp(sha[i], regions[i].expected, 32) != 0) {
      Serial.printf("[TIED30M_PACKAGE] FAIL sha256 mismatch region=%s", regions[i].label);
      print_sha_prefix("got", sha[i]);
      Serial.println();
      return false;
    }
  }
  Serial.printf("[TIED30M_PACKAGE] PASS footer=PGTIEDv1 bytes=%lu",
                static_cast<unsigned long>(tied::kTotalPackBytes));
  for (std::size_t i = 0; i < 3; ++i) print_sha_prefix(regions[i].label, sha[i]);
  Serial.printf(" verify_ms=%.1f\n", static_cast<float>(micros() - start_us) / 1000.0f);
  return true;
}

bool verify_head_cache_crc() {
  static std::uint32_t table[256];
  for (std::uint32_t i = 0; i < 256; ++i) {
    std::uint32_t c = i;
    for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    table[i] = c;
  }
  const std::uint8_t* p = reinterpret_cast<const std::uint8_t*>(lm_head_cache_fp32);
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < lm_head_cache_bytes; ++i) {
    crc = table[(crc ^ p[i]) & 0xFFu] ^ (crc >> 8);
    if ((i & 0xFFFFFu) == 0) yield();
  }
  crc ^= 0xFFFFFFFFu;
  const bool ok = crc == tied::kHeadCacheCrc32;
  Serial.printf("[TIED30M_HEAD_CRC] %s crc32=0x%08lx expected=0x%08lx\n", ok ? "PASS" : "FAIL",
                static_cast<unsigned long>(crc), static_cast<unsigned long>(tied::kHeadCacheCrc32));
  return ok;
}

float l2_norm(const float* values, std::size_t count) { double sum=0.0; for(std::size_t i=0;i<count;++i) sum += static_cast<double>(values[i]) * static_cast<double>(values[i]); return sqrtf(static_cast<float>(sum)); }
float checksum(const float* values, std::size_t count) { double sum=0.0; for(std::size_t i=0;i<count;++i) sum += static_cast<double>(values[i]); return static_cast<float>(sum); }
float max_abs_value(const float* values, std::size_t count) { float r=0.0f; for(std::size_t i=0;i<count;++i){ float a=fabsf(values[i]); if(a>r) r=a;} return r; }
bool all_finite(const float* values, std::size_t count) { for(std::size_t i=0;i<count;++i) if(!isfinite(values[i])) return false; return true; }

void print_ids(const std::uint32_t* ids, std::size_t count) {
  Serial.print("  ids:");
  for (std::size_t i = 0; i < count; ++i) Serial.printf(" %u", static_cast<unsigned>(ids[i]));
  Serial.println();
}

void print_stats(const char* label, const float* values, std::size_t count) {
  Serial.printf("  %-12s l2=%.6f checksum=%.6f max_abs=%.6f finite=%s\n", label, l2_norm(values,count), checksum(values,count), max_abs_value(values,count), all_finite(values,count)?"yes":"NO");
}

void rms_norm(const float* input, const float* weight, float* output, std::size_t count) {
  double sum_sq = 0.0;
  for(std::size_t i=0;i<count;++i) sum_sq += static_cast<double>(input[i]) * static_cast<double>(input[i]);
  const float inv_rms = 1.0f / sqrtf(static_cast<float>(sum_sq / static_cast<double>(count)) + RMS_EPS);
  for(std::size_t i=0;i<count;++i) output[i] = input[i] * inv_rms * weight[i];
}


static inline float q4_lut_load_addx4(
    const float* base,
    std::uint32_t index) {

#if defined(__XTENSA__)

  std::uintptr_t addr;
  const std::uintptr_t base_addr =
      reinterpret_cast<std::uintptr_t>(base);

  __asm__ __volatile__(
      "addx4 %0, %1, %2"
      : "=r"(addr)
      : "r"(index), "r"(base_addr)
  );

  return *reinterpret_cast<const float*>(addr);

#else

  return base[index];

#endif
}

void q4_g64_matvec_raw_rows_pack32x8ph_reference(
    const cfg::MatrixSpec& m,
    const std::uint8_t* packed_q,
    const std::uint16_t* scales_fp16,
    const float* x,
    float* y,
    std::size_t row_begin,
    std::size_t row_end) {

  const std::size_t groups_per_row =
      m.cols / cfg::kGroupSize;

  const std::size_t bytes_per_row =
      m.cols / 2;

  constexpr std::size_t kPackedPerGroup = 32;

                                                               
  static constexpr std::int8_t kSignedInt4[16] = {
     0,  1,  2,  3,
     4,  5,  6,  7,
    -8, -7, -6, -5,
    -4, -3, -2, -1
  };

                            
  alignas(16) float scaled_lut[16];

                                                           
                                                               
  const float* const matrix_scales_fp32 =
      q4_scales_cache_fp32
      + (
          static_cast<std::size_t>(m.scales_offset)
          / sizeof(std::uint16_t)
        );

  for (
      std::size_t row = row_begin;
      row < row_end;
      ++row
  ) {

    const std::uint8_t* row_q =
        packed_q
        + row * bytes_per_row;

    const float* const row_scales_fp32 =
        matrix_scales_fp32
        + row * groups_per_row;

    float acc = 0.0f;

    for (
        std::size_t group = 0;
        group < groups_per_row;
        ++group
    ) {

      const float scale =
          row_scales_fp32[group];

                                                 
                                               
      
      scaled_lut[0]  = 0.0f;
      scaled_lut[1]  = scale;
      scaled_lut[2]  = scale + scale;
      scaled_lut[3]  = scaled_lut[2] + scale;
      scaled_lut[4]  = scaled_lut[2] + scaled_lut[2];
      scaled_lut[5]  = scaled_lut[4] + scale;
      scaled_lut[6]  = scaled_lut[4] + scaled_lut[2];
      scaled_lut[7]  = scaled_lut[6] + scale;

      const float eight_scale =
          scaled_lut[4] + scaled_lut[4];

      scaled_lut[8]  = -eight_scale;
      scaled_lut[9]  = -scaled_lut[7];
      scaled_lut[10] = -scaled_lut[6];
      scaled_lut[11] = -scaled_lut[5];
      scaled_lut[12] = -scaled_lut[4];
      scaled_lut[13] = -scaled_lut[3];
      scaled_lut[14] = -scaled_lut[2];
      scaled_lut[15] = -scale;


      const std::uint8_t* group_q =
          row_q
          + group * kPackedPerGroup;

      const std::size_t col_base =
          group * cfg::kGroupSize;

                    
                                                          
                                         
                                                     
                                           

      const auto* group_q32 =
          static_cast<const std::uint32_t*>(
              __builtin_assume_aligned(group_q, 4)
          );

#define ACC_WORD(W, C) \
      acc += q4_lut_load_addx4(scaled_lut, ((W) >>  0) & 0x0Fu) * x[(C) + 0]; \
      acc += q4_lut_load_addx4(scaled_lut, ((W) >>  4) & 0x0Fu) * x[(C) + 1]; \
      acc += q4_lut_load_addx4(scaled_lut, ((W) >>  8) & 0x0Fu) * x[(C) + 2]; \
      acc += q4_lut_load_addx4(scaled_lut, ((W) >> 12) & 0x0Fu) * x[(C) + 3]; \
      acc += q4_lut_load_addx4(scaled_lut, ((W) >> 16) & 0x0Fu) * x[(C) + 4]; \
      acc += q4_lut_load_addx4(scaled_lut, ((W) >> 20) & 0x0Fu) * x[(C) + 5]; \
      acc += q4_lut_load_addx4(scaled_lut, ((W) >> 24) & 0x0Fu) * x[(C) + 6]; \
      acc += q4_lut_load_addx4(scaled_lut, ((W) >> 28) & 0x0Fu) * x[(C) + 7];

                 
      {
        const std::uint32_t w0 = group_q32[0];
        const std::uint32_t w1 = group_q32[1];
        const std::uint32_t w2 = group_q32[2];
        const std::uint32_t w3 = group_q32[3];

        ACC_WORD(w0, col_base + 0)
        ACC_WORD(w1, col_base + 8)
        ACC_WORD(w2, col_base + 16)
        ACC_WORD(w3, col_base + 24)
      }

                 
      {
        const std::uint32_t w0 = group_q32[4];
        const std::uint32_t w1 = group_q32[5];
        const std::uint32_t w2 = group_q32[6];
        const std::uint32_t w3 = group_q32[7];

        ACC_WORD(w0, col_base + 32)
        ACC_WORD(w1, col_base + 40)
        ACC_WORD(w2, col_base + 48)
        ACC_WORD(w3, col_base + 56)
      }

#undef ACC_WORD

    }

    y[row] = acc;
  }
}


__attribute__((noinline))
void q4_g64_matvec_raw_rows(
    const cfg::MatrixSpec& m,
    const std::uint8_t* packed_q,
    const std::uint16_t* scales_fp16,
    const float* x,
    float* y,
    std::size_t row_begin,
    std::size_t row_end) {

                                                                   
                                               
  if (((row_end - row_begin) & 1u) != 0u) {
    q4_g64_matvec_raw_rows_pack32x8ph_reference(
        m,
        packed_q,
        scales_fp16,
        x,
        y,
        row_begin,
        row_end
    );
    return;
  }

  (void)scales_fp16;

  const std::size_t groups_per_row =
      m.cols / cfg::kGroupSize;

  const std::size_t bytes_per_row =
      m.cols / 2;

  constexpr std::size_t kPackedPerGroup = 32;

  alignas(16) float lut0[16];
  alignas(16) float lut1[16];

  const float* const matrix_scales_fp32 =
      q4_scales_cache_fp32
      + (
          static_cast<std::size_t>(m.scales_offset)
          / sizeof(std::uint16_t)
        );

#define BUILD_LUT(LUT, SCALE) \
  do { \
    (LUT)[0] = 0.0f; \
    (LUT)[1] = (SCALE); \
    (LUT)[2] = (SCALE) + (SCALE); \
    (LUT)[3] = (LUT)[2] + (SCALE); \
    (LUT)[4] = (LUT)[2] + (LUT)[2]; \
    (LUT)[5] = (LUT)[4] + (SCALE); \
    (LUT)[6] = (LUT)[4] + (LUT)[2]; \
    (LUT)[7] = (LUT)[6] + (SCALE); \
    const float eight_scale__ = \
        (LUT)[4] + (LUT)[4]; \
    (LUT)[8]  = -eight_scale__; \
    (LUT)[9]  = -(LUT)[7]; \
    (LUT)[10] = -(LUT)[6]; \
    (LUT)[11] = -(LUT)[5]; \
    (LUT)[12] = -(LUT)[4]; \
    (LUT)[13] = -(LUT)[3]; \
    (LUT)[14] = -(LUT)[2]; \
    (LUT)[15] = -(SCALE); \
  } while (0)


  for (
      std::size_t row = row_begin;
      row < row_end;
      row += 2
  ) {

    const std::size_t row0 = row;
    const std::size_t row1 = row + 1;

    const std::uint8_t* row_q0 =
        packed_q + row0 * bytes_per_row;

    const std::uint8_t* row_q1 =
        packed_q + row1 * bytes_per_row;

    const float* const row_scales0 =
        matrix_scales_fp32
        + row0 * groups_per_row;

    const float* const row_scales1 =
        matrix_scales_fp32
        + row1 * groups_per_row;

    float acc0 = 0.0f;
    float acc1 = 0.0f;


    for (
        std::size_t group = 0;
        group < groups_per_row;
        ++group
    ) {

      const float scale0 = row_scales0[group];
      const float scale1 = row_scales1[group];

      BUILD_LUT(lut0, scale0);
      BUILD_LUT(lut1, scale1);

      const auto* q0 =
          static_cast<const std::uint32_t*>(
              __builtin_assume_aligned(
                  row_q0 + group * kPackedPerGroup,
                  4
              )
          );

      const auto* q1 =
          static_cast<const std::uint32_t*>(
              __builtin_assume_aligned(
                  row_q1 + group * kPackedPerGroup,
                  4
              )
          );

      const std::size_t col_base =
          group * cfg::kGroupSize;


#define ACC_PAIR_ONE(W0, W1, SHIFT, COL) \
      do { \
        const float xv__ = x[(COL)]; \
        const std::uint32_t i0__ = \
            ((W0) >> (SHIFT)) & 0x0Fu; \
        const std::uint32_t i1__ = \
            ((W1) >> (SHIFT)) & 0x0Fu; \
        acc0 += \
            q4_lut_load_addx4(lut0, i0__) * xv__; \
        acc1 += \
            q4_lut_load_addx4(lut1, i1__) * xv__; \
      } while (0)

#define ACC_PAIR_WORD(W0, W1, BASECOL) \
      ACC_PAIR_ONE(W0, W1,  0, (BASECOL) + 0); \
      ACC_PAIR_ONE(W0, W1,  4, (BASECOL) + 1); \
      ACC_PAIR_ONE(W0, W1,  8, (BASECOL) + 2); \
      ACC_PAIR_ONE(W0, W1, 12, (BASECOL) + 3); \
      ACC_PAIR_ONE(W0, W1, 16, (BASECOL) + 4); \
      ACC_PAIR_ONE(W0, W1, 20, (BASECOL) + 5); \
      ACC_PAIR_ONE(W0, W1, 24, (BASECOL) + 6); \
      ACC_PAIR_ONE(W0, W1, 28, (BASECOL) + 7);


      {
        const std::uint32_t w0 = q0[0];
        const std::uint32_t w1 = q1[0];
        ACC_PAIR_WORD(w0, w1, col_base + 0)
      }

      {
        const std::uint32_t w0 = q0[1];
        const std::uint32_t w1 = q1[1];
        ACC_PAIR_WORD(w0, w1, col_base + 8)
      }

      {
        const std::uint32_t w0 = q0[2];
        const std::uint32_t w1 = q1[2];
        ACC_PAIR_WORD(w0, w1, col_base + 16)
      }

      {
        const std::uint32_t w0 = q0[3];
        const std::uint32_t w1 = q1[3];
        ACC_PAIR_WORD(w0, w1, col_base + 24)
      }

      {
        const std::uint32_t w0 = q0[4];
        const std::uint32_t w1 = q1[4];
        ACC_PAIR_WORD(w0, w1, col_base + 32)
      }

      {
        const std::uint32_t w0 = q0[5];
        const std::uint32_t w1 = q1[5];
        ACC_PAIR_WORD(w0, w1, col_base + 40)
      }

      {
        const std::uint32_t w0 = q0[6];
        const std::uint32_t w1 = q1[6];
        ACC_PAIR_WORD(w0, w1, col_base + 48)
      }

      {
        const std::uint32_t w0 = q0[7];
        const std::uint32_t w1 = q1[7];
        ACC_PAIR_WORD(w0, w1, col_base + 56)
      }

#undef ACC_PAIR_WORD
#undef ACC_PAIR_ONE
    }

    y[row0] = acc0;
    y[row1] = acc1;
  }

#undef BUILD_LUT
}


void q4_g64_matvec_raw(
    const cfg::MatrixSpec& m,
    const std::uint8_t* packed_q,
    const std::uint16_t* scales_fp16,
    const float* x,
    float* y) {

  q4_g64_matvec_raw_rows(
      m,
      packed_q,
      scales_fp16,
      x,
      y,
      0,
      m.rows
  );
}



void q4_g64_matvec(
    const MatrixBuf& mb,
    const float* x,
    float* y) {

  q4_g64_matvec_raw(
      mb.spec,
      mb.q4,
      mb.scales,
      x,
      y
  );
}


float silu(float x) { return x / (1.0f + expf(-x)); }

void prepare_rope_cache(
    std::size_t position) {

  if (g_rope_cached_position == position) {
    ++g_rope_cache_hits;
    return;
  }

  const std::uint32_t start_us =
      micros();

  std::size_t pair = 0;

  for (
      std::size_t i = 0;
      i + 1 < cfg::kHeadDim;
      i += 2, ++pair
  ) {
                                                       
                                                       
                                  
    const float inv_freq =
        1.0f
        / powf(
            ROPE_THETA,
            static_cast<float>(i)
            / static_cast<float>(
                cfg::kHeadDim));

    const float angle =
        static_cast<float>(position)
        * inv_freq;

    rope_cos_cache[pair] =
        cosf(angle);

    rope_sin_cache[pair] =
        sinf(angle);
  }

  g_rope_cached_position =
      position;

  ++g_rope_cache_misses;

  g_rope_cache_build_us +=
      micros() - start_us;
}


void apply_rope_one_cached(
    float* vec) {

  std::size_t pair = 0;

  for (
      std::size_t i = 0;
      i + 1 < cfg::kHeadDim;
      i += 2, ++pair
  ) {
    const float c =
        rope_cos_cache[pair];

    const float sn =
        rope_sin_cache[pair];

    const float x0 =
        vec[i];

    const float x1 =
        vec[i + 1];

    vec[i] =
        x0 * c - x1 * sn;

    vec[i + 1] =
        x0 * sn + x1 * c;
  }
}


void apply_rope_qk(
    std::size_t position) {

  prepare_rope_cache(
      position
  );

  for (
      std::size_t h = 0;
      h < cfg::kNHeads;
      ++h
  ) {
    apply_rope_one_cached(
        q_vec
        + h * cfg::kHeadDim
    );
  }

  for (
      std::size_t h = 0;
      h < cfg::kNKvHeads;
      ++h
  ) {
    apply_rope_one_cached(
        k_vec
        + h * cfg::kHeadDim
    );
  }
}


std::size_t kv_offset(std::size_t layer_id, std::size_t position, std::size_t kv_head) {
  return ((layer_id * MAX_PROMPT_TOKENS + position) * cfg::kKvDim) + kv_head * cfg::kHeadDim;
}

bool allocate_kv_cache() {
  const std::size_t values = cfg::kNumLayers * MAX_PROMPT_TOKENS * cfg::kKvDim;
  const std::size_t bytes = values * sizeof(float);
  kv_cache_k = static_cast<float*>(alloc_bytes(bytes, "kv_cache_k"));
  kv_cache_v = static_cast<float*>(alloc_bytes(bytes, "kv_cache_v"));
  if (!kv_cache_k || !kv_cache_v) return false;
  std::memset(kv_cache_k, 0, bytes);
  std::memset(kv_cache_v, 0, bytes);
  return true;
}

void store_current_kv(std::size_t layer_id, std::size_t position) {
  for (std::size_t kv_h = 0; kv_h < cfg::kNKvHeads; ++kv_h) {
    const std::size_t off = kv_offset(layer_id, position, kv_h);
    std::memcpy(kv_cache_k + off, k_vec + kv_h * cfg::kHeadDim, cfg::kHeadDim * sizeof(float));
    std::memcpy(kv_cache_v + off, v_vec + kv_h * cfg::kHeadDim, cfg::kHeadDim * sizeof(float));
  }
}

void attention_from_cache_heads(
    std::size_t layer_id,
    std::size_t position,
    std::size_t head_begin,
    std::size_t head_end,
    float* scores) {

  const float inv_sqrt_hd =
      1.0f / sqrtf(
          static_cast<float>(
              cfg::kHeadDim));

  for (
      std::size_t h = head_begin;
      h < head_end;
      ++h
  ) {
    const std::size_t kv_h =
        h / cfg::kKvRepeat;

    const float* qh =
        q_vec + h * cfg::kHeadDim;

    float max_score =
        -INFINITY;

    for (
        std::size_t t = 0;
        t <= position;
        ++t
    ) {
      const float* kh =
          kv_cache_k
          + kv_offset(
              layer_id,
              t,
              kv_h);

      float score = 0.0f;

      for (
          std::size_t d = 0;
          d < cfg::kHeadDim;
          ++d
      ) {
        score +=
            qh[d] * kh[d];
      }

      score *=
          inv_sqrt_hd;

      scores[t] =
          score;

      if (score > max_score) {
        max_score = score;
      }
    }

    float denom = 0.0f;

    for (
        std::size_t t = 0;
        t <= position;
        ++t
    ) {
      const float e =
          expf(
              scores[t]
              - max_score);

      scores[t] =
          e;

      denom +=
          e;
    }

    if (
        !(denom > 0.0f)
        || !isfinite(denom)
    ) {
      denom = 1.0f;
    }

    float* out =
        attention
        + h * cfg::kHeadDim;

    for (
        std::size_t d = 0;
        d < cfg::kHeadDim;
        ++d
    ) {
      out[d] = 0.0f;
    }

    for (
        std::size_t t = 0;
        t <= position;
        ++t
    ) {
      const float prob =
          scores[t] / denom;

      const float* vh =
          kv_cache_v
          + kv_offset(
              layer_id,
              t,
              kv_h);

      for (
          std::size_t d = 0;
          d < cfg::kHeadDim;
          ++d
      ) {
        out[d] +=
            prob * vh[d];
      }
    }
  }
}


                                                    
void attention_from_cache(
    std::size_t layer_id,
    std::size_t position) {

  attention_from_cache_heads(
      layer_id,
      position,
      0,
      cfg::kNHeads,
      attn_scores
  );
}


void gateup_worker_main(void*) {
  for (;;) {
    ulTaskNotifyTake(
        pdTRUE,
        portMAX_DELAY
    );

    __sync_synchronize();

    const std::uint32_t start_us =
        micros();

                                       
    if (g_gateup_job.attention_job) {

      attention_from_cache_heads(
          g_gateup_job.attention_layer_id,
          g_gateup_job.attention_position,
          g_gateup_job.row_begin,
          g_gateup_job.row_end,
          attn_scores_worker
      );

                                     
      
                                                     
                       
                         
    } else if (g_gateup_job.spec == nullptr) {

      for (
          std::size_t i = g_gateup_job.row_begin;
          i < g_gateup_job.row_end;
          ++i
      ) {
        g_gateup_job.output[i] =
            silu(g_gateup_job.output[i])
            * g_gateup_job.input[i];
      }

    } else if (g_gateup_job.row_range) {

      const std::uint8_t* q_ptr =
          g_gateup_job.use_mmap
          ? g_gateup_job.q_ptr
          : g_gateup_job.fallback->q4;

      const std::uint16_t* scales_ptr =
          g_gateup_job.use_mmap
          ? g_gateup_job.scales_ptr
          : g_gateup_job.fallback->scales;

      q4_g64_matvec_raw_rows(
          *g_gateup_job.spec,
          q_ptr,
          scales_ptr,
          g_gateup_job.input,
          g_gateup_job.output,
          g_gateup_job.row_begin,
          g_gateup_job.row_end
      );

    } else if (g_gateup_job.use_mmap) {

      q4_g64_matvec_raw(
          *g_gateup_job.spec,
          g_gateup_job.q_ptr,
          g_gateup_job.scales_ptr,
          g_gateup_job.input,
          g_gateup_job.output
      );

    } else {

      q4_g64_matvec(
          *g_gateup_job.fallback,
          g_gateup_job.input,
          g_gateup_job.output
      );
    }

    g_gateup_job.elapsed_us =
        micros() - start_us;

    __sync_synchronize();

    xSemaphoreGive(
        g_gateup_done
    );
  }
}


bool init_gateup_worker() {
  g_gateup_done =
      xSemaphoreCreateBinary();

  if (!g_gateup_done) {
    Serial.println(
        "[GATEUP_DUAL] semaphore FAIL"
    );
    return false;
  }

  const BaseType_t rc =
      xTaskCreatePinnedToCore(
          gateup_worker_main,
          "gateup_worker",
          4096,
          nullptr,
          2,
          &g_gateup_worker_task,
          0
      );

  if (rc != pdPASS
      || !g_gateup_worker_task) {

    Serial.println(
        "[GATEUP_DUAL] worker create FAIL"
    );

    vSemaphoreDelete(
        g_gateup_done
    );

    g_gateup_done = nullptr;
    return false;
  }

  Serial.printf(
      "[GATEUP_DUAL] PASS main_core=%d worker_core=0\n",
      xPortGetCoreID()
  );

  return true;
}


bool load_matrix(MatrixBuf& mb, const cfg::MatrixSpec& spec, std::size_t layer_id) {
  if (spec.cols % cfg::kGroupSize != 0 || spec.cols % 2 != 0) {
    Serial.printf("[FAIL] bad matrix shape L%lu.%s rows=%lu cols=%lu group=%lu\n", static_cast<unsigned long>(layer_id), mb.label, static_cast<unsigned long>(spec.rows), static_cast<unsigned long>(spec.cols), static_cast<unsigned long>(cfg::kGroupSize));
    return false;
  }
  if(!ensure_matrix_capacity(mb, spec)) return false;
  mb.spec = spec;
  char label[48]; snprintf(label, sizeof(label), "L%lu.%s", static_cast<unsigned long>(layer_id), mb.label);
  if(!read_exact(part_q4, spec.q4_offset, mb.q4, spec.q4_bytes, label)) return false;
  char scale_label[56]; snprintf(scale_label, sizeof(scale_label), "L%lu.%s.scales", static_cast<unsigned long>(layer_id), mb.label);
  if(!read_exact(part_scales, spec.scales_offset, mb.scales, spec.scales_bytes, scale_label)) return false;
  return true;
}

bool load_layer(std::size_t layer_id){
  const cfg::LayerSpec& layer = cfg::kLayers[layer_id];
  bool ok=true;
  ok = read_fp16_vector(part_kept, layer.attn_norm_kept_offset, attn_norm_weight, cfg::kDModel, "attn_norm") && ok;
  ok = read_fp16_vector(part_kept, layer.ffn_norm_kept_offset, ffn_norm_weight, cfg::kDModel, "ffn_norm") && ok;
  ok = load_matrix(mb_q, layer.q, layer_id) && ok;
  ok = load_matrix(mb_k, layer.k, layer_id) && ok;
  ok = load_matrix(mb_v, layer.v, layer_id) && ok;
  ok = load_matrix(mb_o, layer.o, layer_id) && ok;
  ok = load_matrix(mb_gate, layer.gate, layer_id) && ok;
  ok = load_matrix(mb_up, layer.up, layer_id) && ok;
  ok = load_matrix(mb_down, layer.down, layer_id) && ok;
  return ok;
}

bool run_layer_position(
    std::size_t layer_id,
    std::size_t position,
    std::uint32_t* total_layer_us_out) {

  const cfg::LayerSpec& layer =
      cfg::kLayers[layer_id];

  const std::uint32_t q4_start =
      layer.q.q4_offset;

  const std::uint32_t q4_end =
      layer.down.q4_offset
      + layer.down.q4_bytes;

  const std::uint32_t scales_start =
      layer.q.scales_offset;

  const std::uint32_t scales_end =
      layer.down.scales_offset
      + layer.down.scales_bytes;

  const std::size_t q4_span =
      static_cast<std::size_t>(
          q4_end - q4_start
      );

  const std::size_t scales_span =
      static_cast<std::size_t>(
          scales_end - scales_start
      );

  const void* q4_map = nullptr;
  const void* scales_map = nullptr;

  spi_flash_mmap_handle_t q4_handle = 0;
  spi_flash_mmap_handle_t scales_handle = 0;

  bool q4_mapped = false;
  bool scales_mapped = false;
  bool use_mmap = false;

  const std::uint32_t setup_start_us =
      micros();

  const bool norms_ok =
      read_fp16_vector(
          part_kept,
          layer.attn_norm_kept_offset,
          attn_norm_weight,
          cfg::kDModel,
          "attn_norm"
      )
      &&
      read_fp16_vector(
          part_kept,
          layer.ffn_norm_kept_offset,
          ffn_norm_weight,
          cfg::kDModel,
          "ffn_norm"
      );

  esp_err_t q4_err = ESP_FAIL;
  esp_err_t scales_err = ESP_FAIL;

  if (norms_ok) {
    q4_err =
        esp_partition_mmap(
            part_q4,
            q4_start,
            q4_span,
            ESP_PARTITION_MMAP_DATA,
            &q4_map,
            &q4_handle
        );

    q4_mapped =
        q4_err == ESP_OK;

    if (q4_mapped) {
      scales_err =
          esp_partition_mmap(
              part_scales,
              scales_start,
              scales_span,
              ESP_PARTITION_MMAP_DATA,
              &scales_map,
              &scales_handle
          );

      scales_mapped =
          scales_err == ESP_OK;
    }
  }

  use_mmap =
      norms_ok
      && q4_mapped
      && scales_mapped;

  g_layer_map_us +=
      static_cast<std::uint32_t>(
          micros() - setup_start_us
      );

  if (!use_mmap) {

    if (scales_mapped) {
      spi_flash_munmap(
          scales_handle
      );
      scales_mapped = false;
    }

    if (q4_mapped) {
      spi_flash_munmap(
          q4_handle
      );
      q4_mapped = false;
    }

    ++g_layer_mmap_fallbacks;

    const std::uint32_t load_start_us =
        micros();

    if (!load_layer(layer_id)) {
      return false;
    }

    g_layer_load_us +=
        static_cast<std::uint32_t>(
            micros() - load_start_us
        );
  }

  const std::uint8_t* q4_base =
      static_cast<const std::uint8_t*>(
          q4_map
      );

  const std::uint8_t* scales_base =
      static_cast<const std::uint8_t*>(
          scales_map
      );

  auto matvec = [&](
      const cfg::MatrixSpec& spec,
      const MatrixBuf& fallback,
      const float* input,
      float* output) {

    if (use_mmap) {

      const std::uint8_t* q_ptr =
          q4_base
          + static_cast<std::size_t>(
              spec.q4_offset
              - q4_start
          );

      const std::uint16_t* scales_ptr =
          reinterpret_cast<
              const std::uint16_t*
          >(
              scales_base
              + static_cast<std::size_t>(
                  spec.scales_offset
                  - scales_start
              )
          );

      q4_g64_matvec_raw(
          spec,
          q_ptr,
          scales_ptr,
          input,
          output
      );

    } else {

      q4_g64_matvec(
          fallback,
          input,
          output
      );
    }
  };

  std::uint32_t t0 = micros();

  rms_norm(
      hidden,
      attn_norm_weight,
      normed,
      cfg::kDModel
  );

  const std::uint32_t attn_norm_us =
      micros() - t0;

  const std::size_t q_split_row =
      layer.q.rows / 2;

  const std::uint8_t* q_q_ptr =
      use_mmap
      ? (
          q4_base
          + static_cast<std::size_t>(
              layer.q.q4_offset
              - q4_start
          )
        )
      : mb_q.q4;

  const std::uint16_t* q_scales_ptr =
      use_mmap
      ? reinterpret_cast<const std::uint16_t*>(
          scales_base
          + static_cast<std::size_t>(
              layer.q.scales_offset
              - scales_start
          )
        )
      : mb_q.scales;

  g_gateup_job.use_mmap = use_mmap;
  g_gateup_job.row_range = true;
  g_gateup_job.row_begin = q_split_row;
  g_gateup_job.row_end = layer.q.rows;
  g_gateup_job.spec = &layer.q;
  g_gateup_job.fallback = &mb_q;
  g_gateup_job.q_ptr = q_q_ptr;
  g_gateup_job.scales_ptr = q_scales_ptr;
  g_gateup_job.input = normed;
  g_gateup_job.output = q_vec;
  g_gateup_job.elapsed_us = 0;

  (void)xSemaphoreTake(
      g_gateup_done,
      0
  );

  __sync_synchronize();

  const std::uint32_t q_pair_start_us =
      micros();

  xTaskNotifyGive(
      g_gateup_worker_task
  );

  const std::uint32_t q_main_start_us =
      micros();

  q4_g64_matvec_raw_rows(
      layer.q,
      q_q_ptr,
      q_scales_ptr,
      normed,
      q_vec,
      0,
      q_split_row
  );

  const std::uint32_t q_main_work_us =
      micros() - q_main_start_us;

  if (
      xSemaphoreTake(
          g_gateup_done,
          pdMS_TO_TICKS(5000)
      ) != pdTRUE
  ) {
    Serial.println(
        "[QSPLIT] worker timeout"
    );
    return false;
  }

  __sync_synchronize();

  const std::uint32_t q_pair_us =
      micros() - q_pair_start_us;

  const std::uint32_t q_worker_work_us =
      g_gateup_job.elapsed_us;

  const std::uint32_t q_us =
      q_pair_us;

                                                           
                     
                                               
    
                                         
                                                             
                                                           

  const std::uint8_t* v_q_ptr =
      use_mmap
      ? (
          q4_base
          + static_cast<std::size_t>(
              layer.v.q4_offset
              - q4_start
          )
        )
      : mb_v.q4;

  const std::uint16_t* v_scales_ptr =
      use_mmap
      ? reinterpret_cast<const std::uint16_t*>(
          scales_base
          + static_cast<std::size_t>(
              layer.v.scales_offset
              - scales_start
          )
        )
      : mb_v.scales;

  g_gateup_job.use_mmap =
      use_mmap;

  g_gateup_job.row_range =
      false;

  g_gateup_job.row_begin =
      0;

  g_gateup_job.row_end =
      layer.v.rows;

  g_gateup_job.spec =
      &layer.v;

  g_gateup_job.fallback =
      &mb_v;

  g_gateup_job.q_ptr =
      v_q_ptr;

  g_gateup_job.scales_ptr =
      v_scales_ptr;

  g_gateup_job.input =
      normed;

  g_gateup_job.output =
      v_vec;

  g_gateup_job.elapsed_us = 0;

                                       
  (void)xSemaphoreTake(
      g_gateup_done,
      0
  );

  __sync_synchronize();

  const std::uint32_t kv_pair_start_us =
      micros();

                     
  xTaskNotifyGive(
      g_gateup_worker_task
  );

                                      
  const std::uint32_t k_start_us =
      micros();

  matvec(
      layer.k,
      mb_k,
      normed,
      k_vec
  );

  const std::uint32_t k_us =
      micros() - k_start_us;

                                                          
  if (
      xSemaphoreTake(
          g_gateup_done,
          pdMS_TO_TICKS(5000)
      ) != pdTRUE
  ) {
    Serial.println(
        "[KVDUAL] worker timeout"
    );
    return false;
  }

  __sync_synchronize();

  const std::uint32_t kv_pair_us =
      micros() - kv_pair_start_us;

  const std::uint32_t v_us =
      g_gateup_job.elapsed_us;

  t0 = micros();

                                                            

  const std::uint32_t rope_start_us =
      micros();

  apply_rope_qk(position);

  const std::uint32_t rope_us =
      micros() - rope_start_us;


  const std::uint32_t kvstore_start_us =
      micros();

  store_current_kv(
      layer_id,
      position
  );

  const std::uint32_t kvstore_us =
      micros() - kvstore_start_us;

                                                           
                               
    
                                
    
                                       
                                       
    
                                                 
                                             
                                                           

  const std::size_t attention_head_split =
      cfg::kNHeads / 2;

  g_gateup_job.attention_job =
      true;

  g_gateup_job.attention_layer_id =
      layer_id;

  g_gateup_job.attention_position =
      position;

  g_gateup_job.row_begin =
      attention_head_split;

  g_gateup_job.row_end =
      cfg::kNHeads;

  g_gateup_job.elapsed_us = 0;

                                      
  (void)xSemaphoreTake(
      g_gateup_done,
      0
  );

  __sync_synchronize();

  const std::uint32_t attn_pair_start_us =
      micros();

                          
  xTaskNotifyGive(
      g_gateup_worker_task
  );

                          
  const std::uint32_t attn_main_start_us =
      micros();

  attention_from_cache_heads(
      layer_id,
      position,
      0,
      attention_head_split,
      attn_scores
  );

  const std::uint32_t attn_main_work_us =
      micros() - attn_main_start_us;

  if (
      xSemaphoreTake(
          g_gateup_done,
          pdMS_TO_TICKS(5000)
      ) != pdTRUE
  ) {
    g_gateup_job.attention_job =
        false;

    Serial.println(
        "[ATTNDUAL] worker timeout"
    );

    return false;
  }

  __sync_synchronize();

  const std::uint32_t attn_pair_us =
      micros() - attn_pair_start_us;

  const std::uint32_t attn_worker_work_us =
      g_gateup_job.elapsed_us;

                                                               
  g_gateup_job.attention_job =
      false;

                                                
                                           
  const std::uint32_t attn_us =
      micros() - t0;

                                                           
                                                           
                                                          
                                                           

  const std::size_t o_split_row =
      layer.o.rows / 2;

  const std::uint8_t* o_q_ptr =
      use_mmap
      ? (
          q4_base
          + static_cast<std::size_t>(
              layer.o.q4_offset
              - q4_start
          )
        )
      : mb_o.q4;

  const std::uint16_t* o_scales_ptr =
      use_mmap
      ? reinterpret_cast<const std::uint16_t*>(
          scales_base
          + static_cast<std::size_t>(
              layer.o.scales_offset
              - scales_start
          )
        )
      : mb_o.scales;

  g_gateup_job.use_mmap =
      use_mmap;

  g_gateup_job.row_range =
      true;

  g_gateup_job.row_begin =
      o_split_row;

  g_gateup_job.row_end =
      layer.o.rows;

  g_gateup_job.spec =
      &layer.o;

  g_gateup_job.fallback =
      &mb_o;

  g_gateup_job.q_ptr =
      o_q_ptr;

  g_gateup_job.scales_ptr =
      o_scales_ptr;

  g_gateup_job.input =
      attention;

  g_gateup_job.output =
      o_vec;

  g_gateup_job.elapsed_us = 0;

  (void)xSemaphoreTake(
      g_gateup_done,
      0
  );

  __sync_synchronize();

  const std::uint32_t o_pair_start_us =
      micros();

                            
  xTaskNotifyGive(
      g_gateup_worker_task
  );

                            
  const std::uint32_t o_main_start_us =
      micros();

  q4_g64_matvec_raw_rows(
      layer.o,
      o_q_ptr,
      o_scales_ptr,
      attention,
      o_vec,
      0,
      o_split_row
  );

  const std::uint32_t o_main_work_us =
      micros() - o_main_start_us;

  if (
      xSemaphoreTake(
          g_gateup_done,
          pdMS_TO_TICKS(5000)
      ) != pdTRUE
  ) {
    Serial.println(
        "[OSPLIT] worker timeout"
    );
    return false;
  }

  __sync_synchronize();

  const std::uint32_t o_pair_us =
      micros() - o_pair_start_us;

  const std::uint32_t o_worker_work_us =
      g_gateup_job.elapsed_us;

  const std::uint32_t o_us =
      o_pair_us;

  for (
      std::size_t i = 0;
      i < cfg::kDModel;
      ++i
  ) {
    hidden[i] += o_vec[i];
  }

  t0 = micros();

  rms_norm(
      hidden,
      ffn_norm_weight,
      normed,
      cfg::kDModel
  );

  const std::uint32_t ffn_norm_us =
      micros() - t0;

  const std::uint8_t* up_q_ptr = nullptr;
  const std::uint16_t* up_scales_ptr = nullptr;

  if (use_mmap) {
    up_q_ptr =
        q4_base
        + static_cast<std::size_t>(
            layer.up.q4_offset
            - q4_start
        );

    up_scales_ptr =
        reinterpret_cast<const std::uint16_t*>(
            scales_base
            + static_cast<std::size_t>(
                layer.up.scales_offset
                - scales_start
            )
        );
  }

  g_gateup_job.use_mmap =
      use_mmap;

  g_gateup_job.spec =
      &layer.up;

  g_gateup_job.fallback =
      &mb_up;

  g_gateup_job.q_ptr =
      up_q_ptr;

  g_gateup_job.scales_ptr =
      up_scales_ptr;

  g_gateup_job.input =
      normed;

  g_gateup_job.output =
      up_vec;

  g_gateup_job.row_range = false;
  g_gateup_job.row_begin = 0;
  g_gateup_job.row_end = layer.up.rows;
  g_gateup_job.elapsed_us = 0;

                                              
  (void)xSemaphoreTake(
      g_gateup_done,
      0
  );

  __sync_synchronize();

  const std::uint32_t gateup_start_us =
      micros();

                      
  xTaskNotifyGive(
      g_gateup_worker_task
  );

                                         
  const std::uint32_t gate_start_us =
      micros();

  matvec(
      layer.gate,
      mb_gate,
      normed,
      gate_vec
  );

  const std::uint32_t gate_us =
      micros() - gate_start_us;

                                             
  if (
      xSemaphoreTake(
          g_gateup_done,
          pdMS_TO_TICKS(5000)
      ) != pdTRUE
  ) {
    Serial.println(
        "[GATEUP_DUAL] worker timeout"
    );
    return false;
  }

  __sync_synchronize();

  const std::uint32_t gateup_pair_us =
      micros() - gateup_start_us;

  const std::uint32_t up_us =
      g_gateup_job.elapsed_us;

                                                           
                                    
    
                
    
                       
                        
    
                                                          
                                                           
                                                           

  const std::size_t swiglu_split =
      cfg::kDff / 2;

                                                
    
                                                      
                                         
  g_gateup_job.use_mmap =
      false;

  g_gateup_job.row_range =
      false;

  g_gateup_job.row_begin =
      swiglu_split;

  g_gateup_job.row_end =
      cfg::kDff;

  g_gateup_job.spec =
      nullptr;

  g_gateup_job.fallback =
      nullptr;

  g_gateup_job.q_ptr =
      nullptr;

  g_gateup_job.scales_ptr =
      nullptr;

  g_gateup_job.input =
      up_vec;

  g_gateup_job.output =
      gate_vec;

  g_gateup_job.elapsed_us = 0;

                                          
  (void)xSemaphoreTake(
      g_gateup_done,
      0
  );

  __sync_synchronize();

  const std::uint32_t swiglu_pair_start_us =
      micros();

                                
  xTaskNotifyGive(
      g_gateup_worker_task
  );

                             
  const std::uint32_t swiglu_main_start_us =
      micros();

  for (
      std::size_t i = 0;
      i < swiglu_split;
      ++i
  ) {
    gate_vec[i] =
        silu(gate_vec[i])
        * up_vec[i];
  }

  const std::uint32_t swiglu_main_work_us =
      micros() - swiglu_main_start_us;

  if (
      xSemaphoreTake(
          g_gateup_done,
          pdMS_TO_TICKS(5000)
      ) != pdTRUE
  ) {
    Serial.println(
        "[SWIGLUDUAL] worker timeout"
    );
    return false;
  }

  __sync_synchronize();

  const std::uint32_t swiglu_pair_us =
      micros() - swiglu_pair_start_us;

  const std::uint32_t swiglu_worker_work_us =
      g_gateup_job.elapsed_us;

                                                   
                                      
  const std::uint32_t swiglu_us =
      swiglu_pair_us;

                                                           
                                                          
    
                                                   
                                                          
                                                           

  const std::size_t down_split_row =
      layer.down.rows / 2;

  const std::uint8_t* down_q_ptr =
      use_mmap
      ? (
          q4_base
          + static_cast<std::size_t>(
              layer.down.q4_offset
              - q4_start
          )
        )
      : mb_down.q4;

  const std::uint16_t* down_scales_ptr =
      use_mmap
      ? reinterpret_cast<const std::uint16_t*>(
          scales_base
          + static_cast<std::size_t>(
              layer.down.scales_offset
              - scales_start
          )
        )
      : mb_down.scales;

  g_gateup_job.use_mmap =
      use_mmap;

  g_gateup_job.row_range =
      true;

  g_gateup_job.row_begin =
      down_split_row;

  g_gateup_job.row_end =
      layer.down.rows;

  g_gateup_job.spec =
      &layer.down;

  g_gateup_job.fallback =
      &mb_down;

  g_gateup_job.q_ptr =
      down_q_ptr;

  g_gateup_job.scales_ptr =
      down_scales_ptr;

  g_gateup_job.input =
      gate_vec;

  g_gateup_job.output =
      down_vec;

  g_gateup_job.elapsed_us = 0;

                                          
  (void)xSemaphoreTake(
      g_gateup_done,
      0
  );

  __sync_synchronize();

  const std::uint32_t down_pair_start_us =
      micros();

                                                   
  xTaskNotifyGive(
      g_gateup_worker_task
  );

                                    
  const std::uint32_t down_main_start_us =
      micros();

  q4_g64_matvec_raw_rows(
      layer.down,
      down_q_ptr,
      down_scales_ptr,
      gate_vec,
      down_vec,
      0,
      down_split_row
  );

  const std::uint32_t down_main_work_us =
      micros() - down_main_start_us;

  if (
      xSemaphoreTake(
          g_gateup_done,
          pdMS_TO_TICKS(5000)
      ) != pdTRUE
  ) {
    Serial.println(
        "[DOWNSPLIT] worker timeout"
    );
    return false;
  }

  __sync_synchronize();

  const std::uint32_t down_pair_us =
      micros() - down_pair_start_us;

  const std::uint32_t down_worker_work_us =
      g_gateup_job.elapsed_us;

                                                          
                                                     
  const std::uint32_t down_us =
      down_pair_us;

  for (
      std::size_t i = 0;
      i < cfg::kDModel;
      ++i
  ) {
    hidden[i] += down_vec[i];
  }

  g_attn_norm_us += attn_norm_us;
  g_q_us += q_us;
  g_q_pair_us += q_pair_us;
  g_q_main_work_us += q_main_work_us;
  g_q_worker_work_us += q_worker_work_us;
  g_k_us += k_us;
  g_v_us += v_us;
  g_kv_pair_us += kv_pair_us;
  g_attn_us += attn_us;
  g_rope_us += rope_us;
  g_kvstore_us += kvstore_us;
  g_attn_pair_us += attn_pair_us;
  g_attn_main_work_us += attn_main_work_us;
  g_attn_worker_work_us += attn_worker_work_us;
  g_o_us += o_us;
  g_o_pair_us += o_pair_us;
  g_o_main_work_us += o_main_work_us;
  g_o_worker_work_us += o_worker_work_us;
  g_ffn_norm_us += ffn_norm_us;
  g_gate_us += gate_us;
  g_up_us += up_us;
  g_gateup_pair_us += gateup_pair_us;
  g_swiglu_us += swiglu_us;
  g_swiglu_pair_us += swiglu_pair_us;
  g_swiglu_main_work_us += swiglu_main_work_us;
  g_swiglu_worker_work_us += swiglu_worker_work_us;
  g_down_us += down_us;
  g_down_pair_us += down_pair_us;
  g_down_main_work_us += down_main_work_us;
  g_down_worker_work_us += down_worker_work_us;

  const std::uint32_t total_us =
      attn_norm_us
      + q_us
      + kv_pair_us
      + attn_us
      + o_us
      + ffn_norm_us
      + gateup_pair_us
      + swiglu_us
      + down_us;

  if (total_layer_us_out) {
    *total_layer_us_out =
        total_us;
  }

  if (use_mmap) {

    const std::uint32_t unmap_start_us =
        micros();

    spi_flash_munmap(
        scales_handle
    );

    spi_flash_munmap(
        q4_handle
    );

    g_layer_map_us +=
        static_cast<std::uint32_t>(
            micros() - unmap_start_us
        );
  }

  yield();

  return
      all_finite(
          hidden,
          cfg::kDModel
      )
      &&
      l2_norm(
          hidden,
          cfg::kDModel
      ) > 0.0001f;
}


void insert_topk(std::uint32_t token, float score, std::uint32_t* top_tokens, float* top_scores){
  for(std::size_t i=0;i<TOP_K;++i){
    if(score>top_scores[i]){
      for(std::size_t j=TOP_K-1;j>i;--j){ top_scores[j]=top_scores[j-1]; top_tokens[j]=top_tokens[j-1]; }
      top_scores[i]=score; top_tokens[i]=token; break;
    }
  }
}

bool scan_lm_head_topk(const float* x, std::uint32_t* top_tokens, float* top_scores, std::uint32_t* scan_us_out){
  for(std::size_t i=0;i<TOP_K;++i){ top_tokens[i]=0; top_scores[i]=-INFINITY; }
  const std::uint32_t start=micros();
  if (lm_head_cache_fp32) {
    static_assert(
        (cfg::kDModel % 8) == 0,
        "kDModel must be divisible by 8");

    for(std::uint32_t tok=0;
        tok<cfg::kVocabSize;
        ++tok) {

      const float* row =
          lm_head_cache_fp32
          + static_cast<std::size_t>(tok)
            * cfg::kDModel;

      float acc0 = 0.0f;
      float acc1 = 0.0f;
      float acc2 = 0.0f;
      float acc3 = 0.0f;
      float acc4 = 0.0f;
      float acc5 = 0.0f;
      float acc6 = 0.0f;
      float acc7 = 0.0f;

      for(std::size_t i=0;
          i<cfg::kDModel;
          i += 8) {

        acc0 += row[i + 0] * x[i + 0];
        acc1 += row[i + 1] * x[i + 1];
        acc2 += row[i + 2] * x[i + 2];
        acc3 += row[i + 3] * x[i + 3];
        acc4 += row[i + 4] * x[i + 4];
        acc5 += row[i + 5] * x[i + 5];
        acc6 += row[i + 6] * x[i + 6];
        acc7 += row[i + 7] * x[i + 7];
      }

      const float score =
          ((acc0 + acc1) + (acc2 + acc3))
          + ((acc4 + acc5) + (acc6 + acc7));

      insert_topk(
          tok,
          score,
          top_tokens,
          top_scores);

      if((tok & 0xFFu)==0) {
        yield();
      }
    }
  } else {
    for(std::uint32_t tok=0; tok<cfg::kVocabSize; ++tok){
      const std::size_t off = tied::kLmHeadKeptOffset + static_cast<std::size_t>(tok)*tied::kLmHeadRowBytes;
      if(!read_exact(part_kept, off, fp16_row, tied::kLmHeadRowBytes, "lm_head_row")) return false;
      float score=0.0f;
      for(std::size_t i=0;i<cfg::kDModel;++i) score += fp16_to_float(fp16_row[i])*x[i];
      insert_topk(tok, score, top_tokens, top_scores);
      if((tok & 0xFFu)==0) yield();
    }
  }
  if(scan_us_out) *scan_us_out=micros()-start;
  return true;
}

bool decode_tokens_to_text(const std::uint32_t* ids, std::size_t count, char* out, std::size_t cap, std::size_t* out_bytes) {
  if (!out || cap == 0) return false;
  out[0] = '\0';
  std::size_t bytes = 0;
  const bool ok = pocketlm::tokenizer_runtime_decode(&tokenizer, ids, count, out, cap - 1, &bytes);
  if (bytes >= cap) bytes = cap - 1;
  out[bytes] = '\0';
  if (out_bytes) *out_bytes = bytes;
  return ok;
}

bool init_tokenizer() {
  const bool fs_ok = SPIFFS.begin(false, "/spiffs", 10, "tokenfs");
  if (!fs_ok) { Serial.println("[FAIL] SPIFFS mount"); return false; }
  File normalizer_file = SPIFFS.open("/tokenizer_normalizer.bin", FILE_READ);
  File encoder_file = SPIFFS.open("/tokenizer_encoder.bin", FILE_READ);
  File decoder_file = SPIFFS.open("/tokenizer_decoder.bin", FILE_READ);
  if (!normalizer_file || !encoder_file || !decoder_file) {
    Serial.println("[FAIL] tokenizer files open fail");
    return false;
  }
  const std::uint32_t start = micros();
  const bool ok = pocketlm::tokenizer_runtime_init(&tokenizer, &normalizer_file, &encoder_file, &decoder_file);
  const std::uint32_t us = micros() - start;
  normalizer_file.close(); encoder_file.close(); decoder_file.close();
  Serial.printf("[TOKENIZER] %s init_ms=%.3f free_psram=%u\n", ok ? "PASS" : "FAIL", static_cast<float>(us) / 1000.0f, static_cast<unsigned>(ESP.getFreePsram()));
  return ok;
}

bool check_model_config(){
  bool ok=true;
  ok = ok && cfg::kNumLayers==12;
  ok = ok && cfg::kDModel==cfg::kNHeads*cfg::kHeadDim;
  ok = ok && cfg::kKvDim==cfg::kNKvHeads*cfg::kHeadDim;
  ok = ok && cfg::kNHeads==cfg::kNKvHeads*cfg::kKvRepeat;
  ok = ok && cfg::kDModel%cfg::kGroupSize==0;
  ok = ok && cfg::kDff%cfg::kGroupSize==0;
  Serial.printf("[CONFIG] %s layers=%lu d_model=%lu d_ff=%lu vocab=%lu\n",
    ok ? "PASS" : "FAIL",
    static_cast<unsigned long>(cfg::kNumLayers), static_cast<unsigned long>(cfg::kDModel),
    static_cast<unsigned long>(cfg::kDff), static_cast<unsigned long>(cfg::kVocabSize));
  return ok;
}

bool run_model_token(std::uint32_t tok, std::size_t pos, const char* phase, std::uint32_t* layers_us_out) {
  (void)phase;
  if (pos >= MAX_PROMPT_TOKENS) {
    Serial.printf("[FAIL] context position out of range pos=%u max=%u\n", static_cast<unsigned>(pos), static_cast<unsigned>(MAX_PROMPT_TOKENS));
    return false;
  }
  if (!read_embedding_row(tok, hidden)) {
    Serial.printf("[FAIL] embedding read token=%u pos=%u\n", static_cast<unsigned>(tok), static_cast<unsigned>(pos));
    return false;
  }
  std::uint32_t token_layers_us = 0;
  for (std::size_t layer_id = 0; layer_id < cfg::kNumLayers; ++layer_id) {
    std::uint32_t layer_us = 0;
    if (!run_layer_position(layer_id, pos, &layer_us)) {
      Serial.printf("[FAIL] layer %lu failed at pos=%u\n", static_cast<unsigned long>(layer_id), static_cast<unsigned>(pos));
      return false;
    }
    token_layers_us += layer_us;
    yield();
  }
  if (layers_us_out) *layers_us_out = token_layers_us;
  return all_finite(hidden, cfg::kDModel);
}

bool predict_next_token(std::size_t step, std::uint32_t* next_token, std::uint32_t* lm_head_us_out) {
  (void)step;
  if(!read_fp16_vector(part_kept, cfg::kFinalNormKeptOffset, final_norm_weight, cfg::kDModel, "final_norm")) return false;
  rms_norm(hidden, final_norm_weight, final_normed, cfg::kDModel);

  std::uint32_t top_tokens[TOP_K];
  float top_scores[TOP_K];
  std::uint32_t lm_head_us = 0;
  if(!scan_lm_head_topk(final_normed, top_tokens, top_scores, &lm_head_us)) return false;
  if(lm_head_us_out) *lm_head_us_out = lm_head_us;

  const bool ok = all_finite(final_normed, cfg::kDModel) && isfinite(top_scores[0]);
  if (!ok) Serial.println("[FAIL] next-token logits invalid");
  if (next_token) *next_token = top_tokens[0];
  return ok;
}


struct ChatResult {
  bool ok;
  bool stopped;
  std::size_t generated_count;
  std::uint32_t run_us;
  std::uint32_t generation_us;
  std::uint32_t layers_us;
  std::uint32_t lm_head_us;
  char raw_text[MAX_DECODE_BYTES];
  char clean_text[MAX_DECODE_BYTES];
};

void reset_kv_cache() {
  const std::size_t values = cfg::kNumLayers * MAX_PROMPT_TOKENS * cfg::kKvDim;
  const std::size_t bytes = values * sizeof(float);
  if (kv_cache_k) std::memset(kv_cache_k, 0, bytes);
  if (kv_cache_v) std::memset(kv_cache_v, 0, bytes);
}

bool has_sentence_ender(const char* text) {
  if (!text) return false;
  for (const char* p = text; *p; ++p) {
    if (*p == '.' || *p == '!' || *p == '?') return true;
  }
  return false;
}

void truncate_after_sentence_ender(char* text) {
  if (!text) return;
  for (char* p = text; *p; ++p) {
    if (*p == '.' || *p == '!' || *p == '?') {
      p[1] = '\0';
      return;
    }
  }
}

void trim_ascii_in_place(char* text) {
  if (!text) return;
  std::size_t n = std::strlen(text);
  while (n > 0 && (text[n - 1] == ' ' || text[n - 1] == '\t' || text[n - 1] == '\r' || text[n - 1] == '\n')) {
    text[n - 1] = '\0';
    --n;
  }
  char* start = text;
  while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n') ++start;
  if (start != text) std::memmove(text, start, std::strlen(start) + 1);
}


                                                               
                                      
  
                                                                  
  
                              
                                     
                                     
                                     
                                   
                                                     
                               
                                                               

bool final_is_hspace(char c) {
  return c == ' ' || c == '\t';
}

bool final_is_digit(char c) {
  return c >= '0' && c <= '9';
}

bool final_is_math_op(char c) {
  return c == '+' || c == '-' || c == '*' || c == '/';
}

bool final_is_math_span_char(char c) {
  return final_is_digit(c)
      || final_is_hspace(c)
      || c == '.'
      || final_is_math_op(c)
      || c == '('
      || c == ')';
}

bool final_append_char(
    char c,
    char* out,
    std::size_t cap,
    std::size_t* n) {
  if (!out || !n || cap == 0 || *n + 1 >= cap) return false;
  out[(*n)++] = c;
  out[*n] = '\0';
  return true;
}

bool final_replace_math_glyphs(
    const char* input,
    char* out,
    std::size_t cap) {
  if (!input || !out || cap == 0) return false;
  std::size_t n = 0;
  out[0] = '\0';

  const unsigned char* p =
      reinterpret_cast<const unsigned char*>(input);

  while (*p != 0) {
                    
    if (p[0] == 0xC3 && p[1] == 0x97) {
      if (!final_append_char('*', out, cap, &n)) return false;
      p += 2;
      continue;
    }
                    
    if (p[0] == 0xC3 && p[1] == 0xB7) {
      if (!final_append_char('/', out, cap, &n)) return false;
      p += 2;
      continue;
    }
                    
    if (p[0] == 0xC2 && p[1] == 0xB7) {
      if (!final_append_char('*', out, cap, &n)) return false;
      p += 2;
      continue;
    }
                       
    if (p[0] == 0xE2 && p[1] == 0x8B && p[2] == 0x85) {
      if (!final_append_char('*', out, cap, &n)) return false;
      p += 3;
      continue;
    }
                       
    if (p[0] == 0xE2 && p[1] == 0x88 && p[2] == 0x92) {
      if (!final_append_char('-', out, cap, &n)) return false;
      p += 3;
      continue;
    }

    if (!final_append_char(
            static_cast<char>(*p),
            out,
            cap,
            &n)) {
      return false;
    }
    ++p;
  }

  return true;
}

bool final_canonicalize_input(
    const char* input,
    char* out,
    std::size_t cap) {
  if (!input || !out || cap == 0) return false;

  char glyphs[256] = {};
  char collapsed[256] = {};

  if (!final_replace_math_glyphs(
          input,
          glyphs,
          sizeof(glyphs))) {
    return false;
  }

  std::size_t cn = 0;
  bool pending_space = false;

  for (std::size_t i = 0; glyphs[i] != '\0'; ++i) {
    const char c = glyphs[i];

    if (final_is_hspace(c)) {
      if (cn > 0) pending_space = true;
      continue;
    }

    if (pending_space) {
      if (!final_append_char(
              ' ',
              collapsed,
              sizeof(collapsed),
              &cn)) {
        return false;
      }
      pending_space = false;
    }

    if (!final_append_char(
            c,
            collapsed,
            sizeof(collapsed),
            &cn)) {
      return false;
    }
  }

  std::size_t on = 0;
  out[0] = '\0';
  const std::size_t len = std::strlen(collapsed);

  std::size_t i = 0;
  while (i < len) {
    if (final_is_digit(collapsed[i])) {
      std::size_t j = i;
      std::size_t last_digit_plus_one = i + 1;
      bool has_operator = false;

      while (
          j < len
          && final_is_math_span_char(collapsed[j])) {
        if (final_is_digit(collapsed[j])) {
          last_digit_plus_one = j + 1;
        }
        if (final_is_math_op(collapsed[j])) {
          has_operator = true;
        }
        ++j;
      }

      if (
          has_operator
          && last_digit_plus_one > i + 1) {
        for (
            std::size_t k = i;
            k < last_digit_plus_one;
            ++k) {
          if (final_is_hspace(collapsed[k])) continue;

          if (!final_append_char(
                  collapsed[k],
                  out,
                  cap,
                  &on)) {
            return false;
          }
        }

        i = last_digit_plus_one;
        continue;
      }
    }

    if (!final_append_char(
            collapsed[i],
            out,
            cap,
            &on)) {
      return false;
    }

    ++i;
  }

  return true;
}

void clean_generated_text(
    const char* raw,
    char* out,
    std::size_t cap) {
  if (!out || cap == 0) return;
  out[0] = '\0';

  if (!raw) return;

  std::strncpy(
      out,
      raw,
      cap - 1);

  out[cap - 1] = '\0';
  trim_ascii_in_place(out);
}

bool generate_reply_for_user_text(const char* user_text, ChatResult* result) {
  if (!user_text || !result) return false;
  std::memset(result, 0, sizeof(*result));
  g_gateup_pair_us = 0;
  g_down_pair_us = 0;
  g_q_pair_us = 0;
  g_o_pair_us = 0;
  g_kv_pair_us = 0;
  g_o_main_work_us = 0;
  g_o_worker_work_us = 0;
  g_q_main_work_us = 0;
  g_q_worker_work_us = 0;
  g_down_main_work_us = 0;
  g_down_worker_work_us = 0;
  g_layer_load_us = 0;
  g_layer_map_us = 0;
  g_layer_mmap_fallbacks = 0;

  g_attn_norm_us = 0;
  g_q_us = 0;
  g_k_us = 0;
  g_v_us = 0;
  g_attn_us = 0;
  g_rope_us = 0;
  g_rope_cache_build_us = 0;
  g_rope_cache_hits = 0;
  g_rope_cache_misses = 0;
  g_rope_cached_position = static_cast<std::size_t>(-1);
  g_kvstore_us = 0;
  g_attn_pair_us = 0;
  g_attn_main_work_us = 0;
  g_attn_worker_work_us = 0;
  g_o_us = 0;
  g_ffn_norm_us = 0;
  g_gate_us = 0;
  g_up_us = 0;
  g_swiglu_us = 0;
  g_swiglu_pair_us = 0;
  g_swiglu_main_work_us = 0;
  g_swiglu_worker_work_us = 0;
  g_down_us = 0;

  char canonical_text[256] = {};

  if (!final_canonicalize_input(
          user_text,
          canonical_text,
          sizeof(canonical_text))) {
    Serial.println(
        "[CHAT] canonicalization FAIL");
    return false;
  }

  Serial.printf(
      "[CHAT] canonical: [%s]\n",
      canonical_text);

  std::uint32_t question_ids[
      MAX_PROMPT_TOKENS] = {};

  std::uint32_t token_ids[
      MAX_PROMPT_TOKENS] = {};

  std::uint32_t generated[
      MAX_GENERATE_TOKENS] = {};

  std::size_t question_count = 0;
  std::size_t token_count = 0;
  std::size_t generated_count = 0;

  const std::uint32_t enc_start = micros();

  const bool enc_ok =
      pocketlm::tokenizer_runtime_encode(
          &tokenizer,
          canonical_text,
          std::strlen(canonical_text),
          question_ids,
          MAX_PROMPT_TOKENS - 2,
          &question_count);

  const std::uint32_t enc_us =
      micros() - enc_start;

  if (!enc_ok || question_count == 0) {
    Serial.printf(
        "[CHAT] encode FAIL "
        "tokens=%u encode_us=%u\n",
        static_cast<unsigned>(
            question_count),
        static_cast<unsigned>(
            enc_us));
    return false;
  }

  token_ids[token_count++] =
      FINAL_USER_ID;

  for (
      std::size_t i = 0;
      i < question_count;
      ++i) {
    token_ids[token_count++] =
        question_ids[i];
  }

  token_ids[token_count++] =
      FINAL_ASSISTANT_ID;

  Serial.printf(
      "[CHAT] native prompt "
      "question_tokens=%u "
      "total_tokens=%u "
      "encode_us=%u\n",
      static_cast<unsigned>(
          question_count),
      static_cast<unsigned>(
          token_count),
      static_cast<unsigned>(
          enc_us));

  print_ids(
      token_ids,
      token_count);

  if (token_count + MAX_GENERATE_TOKENS >= MAX_PROMPT_TOKENS) {
    Serial.printf("[CHAT] prompt too long tokens=%u max=%u\n", static_cast<unsigned>(token_count), static_cast<unsigned>(MAX_PROMPT_TOKENS));
    return false;
  }

  reset_kv_cache();

  std::uint32_t total_layers_us = 0;
  std::uint32_t total_lm_head_us = 0;
  const std::uint32_t run_start = micros();

  for (std::size_t pos = 0; pos < token_count; ++pos) {
    std::uint32_t token_layers_us = 0;
    if (!run_model_token(token_ids[pos], pos, "PROMPT", &token_layers_us)) {
      Serial.printf("[CHAT] prompt token FAIL pos=%u id=%u\n", static_cast<unsigned>(pos), static_cast<unsigned>(token_ids[pos]));
      return false;
    }
    total_layers_us += token_layers_us;
  }

  const std::uint32_t gen_start = micros();
  std::size_t next_context_pos = token_count;
  bool stopped = false;

  for (std::size_t step = 0; step < MAX_GENERATE_TOKENS; ++step) {
    std::uint32_t next_tok = 0;
    std::uint32_t lm_head_us = 0;
    if (!predict_next_token(step, &next_tok, &lm_head_us)) {
      Serial.printf("[CHAT] predict FAIL step=%u\n", static_cast<unsigned>(step));
      return false;
    }
    total_lm_head_us += lm_head_us;
    if (next_tok == FINAL_END_ID) {
      Serial.printf(
          "[CHAT] END step=%u generated=%u\n",
          static_cast<unsigned>(step),
          static_cast<unsigned>(generated_count));
      stopped = true;
      break;
    }

    generated[generated_count++] =
        next_tok;

    std::size_t gen_bytes = 0;
    const bool dec_ok = decode_tokens_to_text(generated, generated_count, result->raw_text, sizeof(result->raw_text), &gen_bytes);
    if (!dec_ok) {
      Serial.printf("[CHAT] decode FAIL step=%u\n", static_cast<unsigned>(step));
      return false;
    }
    if (
        FINAL_STOP_AT_FIRST_SENTENCE
        && has_sentence_ender(
            result->raw_text)) {
      stopped = true;
      break;
    }

    if (step + 1 < MAX_GENERATE_TOKENS) {
      std::uint32_t token_layers_us = 0;
      if (!run_model_token(next_tok, next_context_pos, "GEN", &token_layers_us)) {
        Serial.printf("[CHAT] generated token feed FAIL pos=%u id=%u\n", static_cast<unsigned>(next_context_pos), static_cast<unsigned>(next_tok));
        return false;
      }
      total_layers_us += token_layers_us;
      ++next_context_pos;
    }
  }

  result->run_us = micros() - run_start;
  result->generation_us = micros() - gen_start;
  result->layers_us = total_layers_us;
  result->lm_head_us = total_lm_head_us;
  result->generated_count = generated_count;
  result->stopped = stopped;
  clean_generated_text(result->raw_text, result->clean_text, sizeof(result->clean_text));
  result->ok = generated_count > 0 && result->clean_text[0] != '\0';
  return result->ok;
}

char input_line[160];
std::size_t input_len = 0;
bool runtime_ready = false;

void print_prompt() {
  Serial.println();
  Serial.print("You> ");
}

void process_user_line(const char* line) {
  if (!line || line[0] == '\0') return;
  if (!runtime_ready) {
    Serial.println("[CHAT] runtime is not ready");
    return;
  }

  Serial.println();
  Serial.printf("You: %s\n", line);
  Serial.println("PocketGPT: thinking... [TIED30M Q4/FP16 dedup]");

  ChatResult result{};
  const bool ok = generate_reply_for_user_text(line, &result);
  if (!ok) {
    if (result.stopped && result.generated_count == 0) {
      Serial.println(
          "PocketGPT: [empty generation: END before first token]");
    } else {
      Serial.println("PocketGPT: [generation failed]");
    }
    print_prompt();
    return;
  }

  Serial.printf("PocketGPT: %s\n", result.clean_text);
  Serial.printf("[RAW] %s\n", result.raw_text);
  const float run_ms = static_cast<float>(result.run_us) / 1000.0f;
  const float gen_ms = static_cast<float>(result.generation_us) / 1000.0f;
  const float gen_tps = result.generation_us > 0 ? static_cast<float>(result.generated_count) / (static_cast<float>(result.generation_us) / 1000000.0f) : 0.0f;
  Serial.printf("[STATS] tokens=%u stopped=%s total_ms=%.1f generation_ms=%.1f gen_tps=%.4f layers_ms=%.1f lm_head_ms=%.1f\n",
    static_cast<unsigned>(result.generated_count),
    result.stopped ? "yes" : "no",
    run_ms,
    gen_ms,
    gen_tps,
    static_cast<float>(result.layers_us) / 1000.0f,
    static_cast<float>(result.lm_head_us) / 1000.0f);

  Serial.printf(
    "[PROFILE] layer_load_ms=%.1f layer_map_ms=%.1f mmap_fallbacks=%lu\n",
    static_cast<float>(g_layer_load_us) / 1000.0f,
    static_cast<float>(g_layer_map_us) / 1000.0f,
    static_cast<unsigned long>(g_layer_mmap_fallbacks)
  );

  Serial.printf(
    "[OPS] normA=%.1f q=%.1f k=%.1f v=%.1f attn=%.1f o=%.1f "
    "normF=%.1f gate=%.1f up=%.1f swiglu=%.1f down=%.1f\n",
    static_cast<float>(g_attn_norm_us) / 1000.0f,
    static_cast<float>(g_q_us) / 1000.0f,
    static_cast<float>(g_k_us) / 1000.0f,
    static_cast<float>(g_v_us) / 1000.0f,
    static_cast<float>(g_attn_us) / 1000.0f,
    static_cast<float>(g_o_us) / 1000.0f,
    static_cast<float>(g_ffn_norm_us) / 1000.0f,
    static_cast<float>(g_gate_us) / 1000.0f,
    static_cast<float>(g_up_us) / 1000.0f,
    static_cast<float>(g_swiglu_us) / 1000.0f,
    static_cast<float>(g_down_us) / 1000.0f
  );

  Serial.printf(
      "[GATEUP] pair_ms=%.1f gate_work_ms=%.1f up_work_ms=%.1f\n",
      static_cast<float>(g_gateup_pair_us) / 1000.0f,
      static_cast<float>(g_gate_us) / 1000.0f,
      static_cast<float>(g_up_us) / 1000.0f
  );


  Serial.printf(
      "[DOWNSPLIT] pair_ms=%.1f main_work_ms=%.1f worker_work_ms=%.1f\n",
      static_cast<float>(g_down_pair_us) / 1000.0f,
      static_cast<float>(g_down_main_work_us) / 1000.0f,
      static_cast<float>(g_down_worker_work_us) / 1000.0f
  );


  Serial.printf(
      "[QSPLIT] pair_ms=%.1f main_work_ms=%.1f worker_work_ms=%.1f\n",
      static_cast<float>(g_q_pair_us) / 1000.0f,
      static_cast<float>(g_q_main_work_us) / 1000.0f,
      static_cast<float>(g_q_worker_work_us) / 1000.0f
  );


  Serial.printf(
      "[OSPLIT] pair_ms=%.1f main_work_ms=%.1f worker_work_ms=%.1f\n",
      static_cast<float>(g_o_pair_us) / 1000.0f,
      static_cast<float>(g_o_main_work_us) / 1000.0f,
      static_cast<float>(g_o_worker_work_us) / 1000.0f
  );

  Serial.printf(
      "[KVDUAL] pair_ms=%.1f k_work_ms=%.1f v_work_ms=%.1f\n",
      static_cast<float>(g_kv_pair_us) / 1000.0f,
      static_cast<float>(g_k_us) / 1000.0f,
      static_cast<float>(g_v_us) / 1000.0f
  );

  Serial.printf(
      "[ATTNDUAL] pair_ms=%.1f main_work_ms=%.1f worker_work_ms=%.1f\n",
      static_cast<float>(g_attn_pair_us) / 1000.0f,
      static_cast<float>(g_attn_main_work_us) / 1000.0f,
      static_cast<float>(g_attn_worker_work_us) / 1000.0f
  );

  Serial.printf(
      "[ROPEKV] rope_ms=%.1f kvstore_ms=%.1f combined_ms=%.1f\n",
      static_cast<float>(g_rope_us) / 1000.0f,
      static_cast<float>(g_kvstore_us) / 1000.0f,
      static_cast<float>(g_rope_us + g_kvstore_us) / 1000.0f
  );

  Serial.printf(
      "[ROPECACHE] build_ms=%.1f hits=%llu misses=%llu\n",
      static_cast<float>(g_rope_cache_build_us) / 1000.0f,
      static_cast<unsigned long long>(g_rope_cache_hits),
      static_cast<unsigned long long>(g_rope_cache_misses)
  );




  Serial.printf(
      "[SWIGLUDUAL] pair_ms=%.1f main_work_ms=%.1f worker_work_ms=%.1f\n",
      static_cast<float>(g_swiglu_pair_us) / 1000.0f,
      static_cast<float>(g_swiglu_main_work_us) / 1000.0f,
      static_cast<float>(g_swiglu_worker_work_us) / 1000.0f
  );



  print_prompt();
}

}             


bool cache_q4_scales_to_fp32_psram() {

  static_assert(
      (cfg::kScalesF16Bytes % sizeof(std::uint16_t)) == 0,
      "scale byte count must be FP16 aligned"
  );

  q4_scales_cache_count =
      static_cast<std::size_t>(
          cfg::kScalesF16Bytes
      )
      / sizeof(std::uint16_t);

  const std::size_t bytes =
      q4_scales_cache_count
      * sizeof(float);

  q4_scales_cache_fp32 =
      static_cast<float*>(
          heap_caps_malloc(
              bytes,
              MALLOC_CAP_SPIRAM
              | MALLOC_CAP_8BIT
          )
      );

  if (!q4_scales_cache_fp32) {
    Serial.printf(
        "[SCALECACHE] FAIL alloc bytes=%lu free_psram=%lu\n",
        static_cast<unsigned long>(bytes),
        static_cast<unsigned long>(
            ESP.getFreePsram()
        )
    );
    return false;
  }

  const std::uint32_t start_us =
      micros();

  std::size_t done = 0;

  while (
      done < q4_scales_cache_count
  ) {

    const std::size_t remaining =
        q4_scales_cache_count - done;

    const std::size_t n =
        remaining < cfg::kDModel
        ? remaining
        : cfg::kDModel;

    if (
        !read_exact(
            part_scales,
            done * sizeof(std::uint16_t),
            fp16_row,
            n * sizeof(std::uint16_t),
            "q4_scale_cache"
        )
    ) {
      heap_caps_free(
          q4_scales_cache_fp32
      );

      q4_scales_cache_fp32 = nullptr;
      q4_scales_cache_count = 0;

      Serial.println(
          "[SCALECACHE] FAIL read"
      );

      return false;
    }

    for (
        std::size_t i = 0;
        i < n;
        ++i
    ) {
      q4_scales_cache_fp32[done + i] =
          fp16_to_float(
              fp16_row[i]
          );
    }

    done += n;
  }

  const std::uint32_t load_us =
      micros() - start_us;

                                                  
  double checksum = 0.0;

  const std::size_t checksum_n =
      q4_scales_cache_count < 4096
      ? q4_scales_cache_count
      : 4096;

  for (
      std::size_t i = 0;
      i < checksum_n;
      ++i
  ) {
    checksum +=
        static_cast<double>(
            q4_scales_cache_fp32[i]
        )
        * static_cast<double>(
            i + 1
        );
  }

  Serial.printf(
      "[SCALECACHE] PASS count=%lu bytes=%lu load_ms=%.3f checksum4096=%.9f free_psram=%lu\n",
      static_cast<unsigned long>(
          q4_scales_cache_count
      ),
      static_cast<unsigned long>(
          bytes
      ),
      static_cast<float>(
          load_us
      ) / 1000.0f,
      checksum,
      static_cast<unsigned long>(
          ESP.getFreePsram()
      )
  );

  return true;
}


void setup(){
  Serial.begin(115200);
  delay(1500);
  Serial.println();
  Serial.println("=== PocketGPT-S3 TIED30M dedup serial chat (30.46M tied, 20.24MB pack) ===");
  Serial.printf("Flash=%lu PSRAM=%s free_psram=%lu\n", static_cast<unsigned long>(ESP.getFlashChipSize()), psramFound()?"yes":"no", static_cast<unsigned long>(ESP.getFreePsram()));

  init_matrix_bufs();
  bool ok = true;
  ok = check_model_config() && ok;
  ok = init_tokenizer() && ok;
  ok = allocate_kv_cache() && ok;
  ok = init_gateup_worker() && ok;
  part_q4 = find_part("v5e_q4", cfg::kWeightsQ4Bytes);
  part_scales = find_part("v5e_scales", cfg::kScalesF16Bytes);
  part_kept = find_part("v5e_kept", tied::kKeptTensorsBytes);
  ok = ok && part_q4 && part_scales && part_kept;
  if (ok) ok = verify_tied30m_package() && ok;
  if (ok) ok = cache_lm_head_to_psram() && ok;
  if (ok) ok = verify_head_cache_crc() && ok;
  if (ok) ok = cache_q4_scales_to_fp32_psram() && ok;

  runtime_ready = ok;
  if (ok) {
    Serial.println("[READY] Local chat demo. Type a short message and press Enter.");
    Serial.println("Stops on END or after 24 generated tokens. Try: hello | who are you? | i feel sad");
    print_prompt();
  } else {
    Serial.println("[FAIL] TIED30M serial chat init failed; inference disabled");
  }
}

void loop(){
  while (Serial.available() > 0) {
    char c = static_cast<char>(Serial.read());
    if (c == '\r') continue;
    if (c == '\n') {
      input_line[input_len] = '\0';
                              
      while (input_len > 0 && (input_line[input_len - 1] == ' ' || input_line[input_len - 1] == '\t')) {
        input_line[--input_len] = '\0';
      }
                                                        
      char* start = input_line;
      while (*start == ' ' || *start == '\t') ++start;
      if (start != input_line) {
        std::memmove(input_line, start, std::strlen(start) + 1);
        input_len = std::strlen(input_line);
      }
      if (input_len > 0) {
        process_user_line(input_line);
      } else {
        print_prompt();
      }
      input_len = 0;
      input_line[0] = '\0';
    } else if (c == 8 || c == 127) {
      if (input_len > 0) --input_len;
      input_line[input_len] = '\0';
    } else if (input_len + 1 < sizeof(input_line)) {
      input_line[input_len++] = c;
      input_line[input_len] = '\0';
    } else {
      Serial.println();
      Serial.println("[CHAT] input buffer full; press Enter to run or reset to clear.");
    }
  }
  delay(10);
}
