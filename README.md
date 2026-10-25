# PocketGPT on ESP32-S3

![PocketGPT handheld device with its touchscreen keyboard and a recorded answer](docs/images/pocketgpt-device.jpg)

A language model trained from scratch and running offline on an ESP32-S3. Use a touchscreen keyboard on XIAO or chat over USB on DevKit.

[Watch the project video](https://youtu.be/LgCNVUndTH8) · [Hardware on Amazon](https://amzn.to/4B00LyJ)

## Run PocketGPT

**The trained weights and tokenizer are available. No training or API key is needed.** Download the source, build the firmware and install the matching model package using the guide for your board:

| Your hardware | Start here |
|---|---|
| XIAO ESP32-S3: 8 MB Flash / 8 MB PSRAM, Seeed Round Display and microSD | [Touchscreen setup](docs/ASSETS.md#xiao-with-touchscreen-keyboard) |
| ESP32-S3 DevKit: 32 MB octal Flash / 16 MB PSRAM | [USB chat setup](docs/ASSETS.md#devkit) |

Start with the shared preparation steps at the top of the [installation guide](docs/ASSETS.md). Model packages are in [Releases](https://github.com/mmm1712/Local_AI_esp32/releases/tag/v0.1.0-model-assets); firmware is built from this repository. The two boards use different packages.

## What can PocketGPT actually do?

This is a personal experiment with limited capabilities. It can produce short greetings and basic conversational replies, but answers can be wrong, repetitive or unrelated. Even basic arithmetic is unreliable.

These are recorded hardware outputs:

| Prompt | Response | Board |
|---|---|---|
| `hello` | `Hello.` | XIAO |
| `how are you?` | `I'm doing okay.` | XIAO |
| `say something nice` | `You're doing better than you think.` | DevKit |

The XIAO tests took about 6.4 seconds for `hello` and 11.8 seconds for `how are you?`. These examples do not guarantee the same quality for other questions. [XIAO recordings](xiao_ai/recorded_results/ui_2026-10-13/serial_results.txt) · [DevKit recording](espdev_ai/recorded_results/2026-10-12_tied30m_dedup_devkit/tests/03.txt)

## Model specifications

| Characteristic | Value |
|---|---|
| Independent trained parameters | 30,460,864 |
| Architecture | Decoder-only Transformer; 12 layers; hidden size 448 |
| Attention | 8 query heads, 2 key/value heads |
| Tokenizer | SentencePiece; 6,144 tokens |
| Runtime limits | 64-token context; up to 24 generated tokens per response |
| Recorded generation speed | DevKit: about 1.41 tokens/s; XIAO: about 0.742 tokens/s |
| Deployment files | DevKit: 20,241,616-byte deduplicated pack. XIAO: 21,717,088 bytes of model weights across Flash, PSRAM-loaded data and microSD; cache and tokenizer files are separate. See [the exact XIAO layout](docs/MODEL_CARD.md#xiao-storage-layout). |

Speeds are individual measurements, not guarantees. The packages use different mixed-precision layouts. See [model and quantization details](docs/MODEL_CARD.md).

### Optional DevKit INT8 SIMD optimization

The ESP32-S3 DevKit supports an optional INT8 SIMD output head for faster inference.

In 15 paired hardware tests, generation speed increased from approximately 1.42 to 1.65 tokens/s, a 16.4% improvement. The optimization uses the existing trained model without retraining.

The standard FP32 output head remains the default. The optimized implementation is available through the separate `local_devkit_int8` build configuration.

See the [performance comparison results](espdev_ai/recorded_results/2026-10-24_int8_head/comparison.json) for the recorded measurements.

## Training

The [original training code and settings](training/README.md) are included. **Base V2** names the second base-model version, trained from random initialization on about 300 million token positions before later refinement. Exact historical retraining requires additional data and records described in that guide.

## Optional: Gemini or Ollama

The [network interface](docs/NETWORK_SETUP.md) uses the same XIAO/display hardware, but runs a different firmware. Answers come from Gemini's servers or Ollama on your PC. This option does not use PocketGPT's weights.

## Serial testing

After installing PocketGPT on a DevKit, follow the [USB test instructions](docs/BUILD.md#devkit-serial-test) to send a prompt and save the result. The test checks for a nonempty reply, a positive token count and no reported inference errors. Passing confirms generation completed, not that the answer is factually correct.

## Reference

- [Contributing and reporting bugs](docs/contributing-guide.md)
- [Build reference](docs/BUILD.md) · [Validation results](docs/RELEASE_STATUS.md)
- [Data sources](docs/DATA_PROVENANCE.md) · [Privacy](docs/PRIVACY.md)
- [MIT license and third-party notices](LICENSE_STATUS.md)
