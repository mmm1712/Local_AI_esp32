    .text
    .align 4

    .global pocketgpt_head_dot_s8
    .type pocketgpt_head_dot_s8,@function


pocketgpt_head_dot_s8:

    entry   a1, 48

    movi    a8, 16
    blt     a5, a8, .bad_len

    extui   a8, a5, 0, 4
    bnez    a8, .bad_len

    movi.n  a14, 0
    wur.accx_0 a14
    wur.accx_1 a14

    ee.vld.128.ip q0, a2, 16
    ee.vld.128.ip q1, a3, 16

    srli    a5, a5, 4

    loopnez a5, .loop_end

        ee.vmulas.s8.accx.ld.ip q1, a3, 16, q0, q1
        ee.vld.128.ip q0, a2, 16

.loop_end:

    rur.accx_0 a14
    s32i    a14, a4, 0

    movi.n  a2, 0
    retw.n

.bad_len:

    movi    a2, -1
    retw.n
