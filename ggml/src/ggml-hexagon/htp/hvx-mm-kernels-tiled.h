#include "q2-k-scalar.h"

// Dynamic quantizers that produce tiled activations

static inline void quantize_block_f32_q8_1_tiled(float * restrict x, uint8_t * restrict y_block) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero = Q6_V_vzero();

    HVX_Vector vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_Vector vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_Vector vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_Vector vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));  // 1.0 / 127.0
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf              = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    const HVX_Vector ones = Q6_Vb_vsplat_R(1);
    HVX_Vector v_sums = Q6_Vw_vrmpy_VbVb(vx_i8, ones);
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 4));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 8));
    v_sums = Q6_Vw_vadd_VwVw(v_sums, Q6_V_vror_VR(v_sums, 16));

    const HVX_Vector v_inv127 = hvx_vec_splat_f32(1.0f / 127.0f);
    HVX_Vector vd0_sf = hvx_vec_mul_f32_f32(vmax0_sf, v_inv127);
    HVX_Vector vd1_sf = hvx_vec_mul_f32_f32(vmax1_sf, v_inv127);
    HVX_Vector vd2_sf = hvx_vec_mul_f32_f32(vmax2_sf, v_inv127);
    HVX_Vector vd3_sf = hvx_vec_mul_f32_f32(vmax3_sf, v_inv127);

    HVX_Vector v_sums_sf = Q6_Vsf_equals_Vw(v_sums);
    HVX_Vector voff0_sf = hvx_vec_mul_f32_f32(vd0_sf, v_sums_sf);
    HVX_Vector voff1_sf = hvx_vec_mul_f32_f32(vd1_sf, Q6_V_vror_VR(v_sums_sf, 32));
    HVX_Vector voff2_sf = hvx_vec_mul_f32_f32(vd2_sf, Q6_V_vror_VR(v_sums_sf, 64));
    HVX_Vector voff3_sf = hvx_vec_mul_f32_f32(vd3_sf, Q6_V_vror_VR(v_sums_sf, 96));

    HVX_Vector voff01_hf = hvx_vec_f32_to_f16(voff0_sf, voff1_sf);
    HVX_Vector voff23_hf = hvx_vec_f32_to_f16(voff2_sf, voff3_sf);

    HVX_Vector r_scale[4] = {
        hvx_vec_repl_f16(vd01_hf),
        hvx_vec_repl_f16(Q6_V_vror_VR(vd01_hf, 64)),
        hvx_vec_repl_f16(vd23_hf),
        hvx_vec_repl_f16(Q6_V_vror_VR(vd23_hf, 64)),
    };
    HVX_Vector r_offset[4] = {
        hvx_vec_repl_f16(voff01_hf),
        hvx_vec_repl_f16(Q6_V_vror_VR(voff01_hf, 64)),
        hvx_vec_repl_f16(voff23_hf),
        hvx_vec_repl_f16(Q6_V_vror_VR(voff23_hf, 64)),
    };

    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;

    for (int b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);

        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1280);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale[b];
        dst[9] = r_offset[b];
    }
}

static inline void quantize_block_f32_q8_0_tiled(float * restrict x, uint8_t * restrict y_block) {
    assert((unsigned long) x % 128 == 0);
    assert((unsigned long) y_block % 128 == 0);

    HVX_Vector * vx = (HVX_Vector *) x;
    HVX_Vector zero   = Q6_V_vzero();

    HVX_Vector vmax0_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[0]));
    HVX_Vector vmax1_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[1]));
    HVX_Vector vmax2_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[2]));
    HVX_Vector vmax3_sf = hvx_vec_reduce_max_f32(hvx_vec_abs_f32(vx[3]));

    HVX_Vector vx0_qf = Q6_Vqf32_vsub_VsfVsf(vx[0], zero);
    HVX_Vector vx1_qf = Q6_Vqf32_vsub_VsfVsf(vx[1], zero);
    HVX_Vector vx2_qf = Q6_Vqf32_vsub_VsfVsf(vx[2], zero);
    HVX_Vector vx3_qf = Q6_Vqf32_vsub_VsfVsf(vx[3], zero);

    HVX_Vector vmax0_qf = Q6_Vqf32_vsub_VsfVsf(vmax0_sf, zero);
    HVX_Vector vmax1_qf = Q6_Vqf32_vsub_VsfVsf(vmax1_sf, zero);
    HVX_Vector vmax2_qf = Q6_Vqf32_vsub_VsfVsf(vmax2_sf, zero);
    HVX_Vector vmax3_qf = Q6_Vqf32_vsub_VsfVsf(vmax3_sf, zero);

    HVX_Vector vmax01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax1_qf, vmax0_qf)));
    HVX_Vector vmax23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vmax3_qf, vmax2_qf)));

    HVX_Vector vx01_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx1_qf, vx0_qf)));
    HVX_Vector vx23_hf = Q6_Vh_vdeal_Vh(Q6_Vhf_equals_Wqf32(Q6_W_vcombine_VV(vx3_qf, vx2_qf)));

    HVX_Vector vd01_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax01_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_Vector vd23_qf16 = Q6_Vqf16_vmpy_VhfVhf(vmax23_hf, Q6_Vh_vsplat_R(0x2008));
    HVX_Vector vd01_hf   = Q6_Vhf_equals_Vqf16(vd01_qf16);
    HVX_Vector vd23_hf   = Q6_Vhf_equals_Vqf16(vd23_qf16);

    HVX_Vector vd01_inv_hf = hvx_vec_inverse_f16(vd01_hf);
    HVX_Vector vd23_inv_hf = hvx_vec_inverse_f16(vd23_hf);
    vx01_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx01_hf, vd01_inv_hf));
    vx23_hf                = Q6_Vhf_equals_Vqf16(Q6_Vqf16_vmpy_VhfVhf(vx23_hf, vd23_inv_hf));

    HVX_Vector vx01_i16 = hvx_vec_i16_from_hf_rnd_sat(vx01_hf);
    HVX_Vector vx23_i16 = hvx_vec_i16_from_hf_rnd_sat(vx23_hf);
    HVX_Vector vx_i8    = Q6_Vb_vpack_VhVh_sat(vx23_i16, vx01_i16);

    HVX_VectorPair vp01 = Q6_W_vshuff_VVR(vd01_hf, vd01_hf, -64);
    HVX_VectorPair vp23 = Q6_W_vshuff_VVR(vd23_hf, vd23_hf, -64);

    static const uint8_t __attribute__((aligned(128))) repl[128] = {
        0x00, 0x00, 0x00, 0x00, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x40, 0x40, 0x40, 0x40, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x20, 0x20, 0x20, 0x20, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
        0x10, 0x10, 0x10, 0x10, 0x04, 0x04, 0x04, 0x04, 0x08, 0x08, 0x08, 0x08, 0x04, 0x04, 0x04, 0x04,
    };
    HVX_Vector v_repl_ctrl = * (const HVX_Vector *) repl;

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        HVX_Vector v_act = Q6_V_vror_VR(vx_i8, b * 32);
        HVX_Vector r_scale;
        if (b == 0) {
            r_scale = Q6_V_lo_W(vp01);
        } else if (b == 1) {
            r_scale = Q6_V_hi_W(vp01);
        } else if (b == 2) {
            r_scale = Q6_V_lo_W(vp23);
        } else {
            r_scale = Q6_V_hi_W(vp23);
        }

        HVX_Vector r0 = Q6_V_vdelta_VV(v_act, v_repl_ctrl);
        HVX_Vector r1 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 4),  v_repl_ctrl);
        HVX_Vector r2 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 8),  v_repl_ctrl);
        HVX_Vector r3 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 12), v_repl_ctrl);
        HVX_Vector r4 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 16), v_repl_ctrl);
        HVX_Vector r5 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 20), v_repl_ctrl);
        HVX_Vector r6 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 24), v_repl_ctrl);
        HVX_Vector r7 = Q6_V_vdelta_VV(Q6_V_vror_VR(v_act, 28), v_repl_ctrl);

        HVX_Vector * restrict dst = (HVX_Vector *) (y_block + b * 1152);
        dst[0] = r0;
        dst[1] = r1;
        dst[2] = r2;
        dst[3] = r3;
        dst[4] = r4;
        dst[5] = r5;
        dst[6] = r6;
        dst[7] = r7;
        dst[8] = r_scale;
    }
}

static void quantize_row_f32_q8_0_tiled(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (k + qk - 1) / qk;

    for (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 1152;
        quantize_block_f32_q8_0_tiled(x + i * qk, y_block);
    }
}

static void quantize_row_f32_q8_1_tiled(float * restrict x, uint8_t * restrict y, uint32_t k) {
    assert(k % 32 == 0);
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (k + qk - 1) / qk;

    for (uint32_t i = 0; i < nb; i++) {
        uint8_t * restrict y_block = y + i * 4 * 1280;
        quantize_block_f32_q8_1_tiled(x + i * qk, y_block);
    }
}

// Dot kernels & helpers that consume tiled activations

static inline HVX_Vector hvx_vec_mul_f16_f16_to_f32_lower32(HVX_Vector v1, HVX_Vector v2) {
#if __HVX_ARCH__ >= 79
    HVX_VectorPair p = Q6_Wsf_vmpy_VhfVhf(v1, v2);
    return Q6_V_lo_W(Q6_W_vshuff_VVR(Q6_V_hi_W(p), Q6_V_lo_W(p), -4));
#else
    HVX_VectorPair p = Q6_Wqf32_vmpy_VhfVhf(v1, v2);
    HVX_Vector hi = Q6_Vsf_equals_Vqf32(Q6_V_hi_W(p));
    HVX_Vector lo = Q6_Vsf_equals_Vqf32(Q6_V_lo_W(p));
    return Q6_V_lo_W(Q6_W_vshuff_VVR(hi, lo, -4));
#endif
}

static inline HVX_Vector unpack_and_interleave_4bit(HVX_Vector v_a, HVX_Vector v_b, HVX_Vector mask_h4) {
    HVX_Vector v_W0 = Q6_V_vand_VV(v_a, mask_h4);
    HVX_Vector v_W1 = Q6_Vub_vlsr_VubR(v_a, 4);
    HVX_Vector v_W2 = Q6_V_vand_VV(v_b, mask_h4);
    HVX_Vector v_W3 = Q6_Vub_vlsr_VubR(v_b, 4);

    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_W1, v_W0, -1);
    HVX_VectorPair v23_pair = Q6_W_vshuff_VVR(v_W3, v_W2, -1);
    HVX_VectorPair v0123_pair = Q6_W_vshuff_VVR(Q6_V_lo_W(v23_pair), Q6_V_lo_W(v01_pair), -2);
    return Q6_V_lo_W(v0123_pair);
}

static inline HVX_VectorPair unpack_and_interleave_4bit_x2(HVX_Vector v_src, HVX_Vector mask_h4) {
    HVX_Vector v_lo = Q6_V_vand_VV(v_src, mask_h4);
    HVX_Vector v_hi = Q6_Vub_vlsr_VubR(v_src, 4);
    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_Vector v01_lo = Q6_V_lo_W(v01_pair);
    HVX_Vector v01_hi = Q6_V_hi_W(v01_pair);

    HVX_Vector v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_Vector v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_Vector v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_Vector v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    return Q6_W_vcombine_VV(v_W1, v_W0);
}

static inline HVX_Vector accum_4bit_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_Vector accum_4bit_32x1_lut(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_VectorPair accum_4bit_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

static inline HVX_VectorPair accum_4bit_32x2_lut(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector mask_h4,
    HVX_Vector lut
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_4bit_x2(vptr[i], mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vlut32_VbVbI(Q6_V_lo_W(v_W_pair), lut, 0);
        HVX_Vector v_W1 = Q6_Vb_vlut32_VbVbI(Q6_V_hi_W(v_W_pair), lut, 0);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

static inline HVX_Vector accum_q8_0_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act
) {
    HVX_Vector v_sum = Q6_V_vzero();
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum = Q6_Vw_vrmpyacc_VwVbVb(v_sum, v_W, v_act[g]);
    }
    return v_sum;
}

static inline HVX_VectorPair accum_q8_0_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    #pragma unroll
    for (int g = 0; g < 8; g++) {
        HVX_Vector v_rot = Q6_V_vror_VR(vptr[g], 64);
        HVX_Vector v_W = Q6_V_lo_W(Q6_W_vshuff_VVR(v_rot, vptr[g], -2));
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W, v_act0[g]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W, v_act1[g]);
    }
    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q5_K: OR 0x10 into every lane of v whose flag j is set in the plane (see HTP_MM_WEIGHT_TILE_SIZE_Q5_K)
static inline HVX_Vector hvx_q5k_or_hibit(HVX_Vector v, HVX_Vector v_plane, int j) {
    HVX_VectorPred q = Q6_Q_vand_VR(v_plane, 0x01010101u << j);
    return Q6_V_vandor_VQR(v, q, 0x10101010);
}

// 5-bit variant: the high bit comes from the plane, see hvx_q5k_or_hibit
static inline HVX_VectorPair unpack_and_interleave_5bit_x2(HVX_Vector v_src, HVX_Vector v_plane, int i, HVX_Vector mask_h4) {
    HVX_Vector v_lo = hvx_q5k_or_hibit(Q6_V_vand_VV(v_src, mask_h4), v_plane, 2 * i);
    HVX_Vector v_hi = hvx_q5k_or_hibit(Q6_Vub_vlsr_VubR(v_src, 4), v_plane, 2 * i + 1);
    HVX_VectorPair v01_pair = Q6_W_vshuff_VVR(v_hi, v_lo, -1);
    HVX_Vector v01_lo = Q6_V_lo_W(v01_pair);
    HVX_Vector v01_hi = Q6_V_hi_W(v01_pair);

    HVX_Vector v23_lo = Q6_V_valign_VVR(v01_hi, v01_lo, 64);
    HVX_Vector v_W0 = Q6_V_lo_W(Q6_W_vshuff_VVR(v23_lo, v01_lo, -2));

    HVX_Vector v67_lo = Q6_V_valign_VVR(v01_lo, v01_hi, 64);
    HVX_Vector v_W1 = Q6_V_lo_W(Q6_W_vshuff_VVR(v67_lo, v01_hi, -2));

    return Q6_W_vcombine_VV(v_W1, v_W0);
}

static inline HVX_Vector accum_5bit_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_plane = vptr[5];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act[i * 2 + 1]);
    }

    return Q6_Vw_vadd_VwVw(v_sum0, v_sum1);
}

static inline HVX_VectorPair accum_5bit_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector i8
) {
    HVX_Vector v_sum0 = Q6_V_vzero();
    HVX_Vector v_sum1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector v_plane = vptr[5];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        HVX_VectorPair v_W_pair = unpack_and_interleave_5bit_x2(vptr[i], v_plane, i, mask_h4);
        HVX_Vector v_W0 = Q6_Vb_vsub_VbVb(Q6_V_lo_W(v_W_pair), i8);
        HVX_Vector v_W1 = Q6_Vb_vsub_VbVb(Q6_V_hi_W(v_W_pair), i8);

        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W0, v_act0[i * 2 + 0]);
        v_sum0 = Q6_Vw_vrmpyacc_VwVbVb(v_sum0, v_W1, v_act0[i * 2 + 1]);

        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W0, v_act1[i * 2 + 0]);
        v_sum1 = Q6_Vw_vrmpyacc_VwVbVb(v_sum1, v_W1, v_act1[i * 2 + 1]);
    }

    return Q6_W_vcombine_VV(v_sum1, v_sum0);
}

// Q6_K weights are stored unsigned (0..63), see HTP_MM_WEIGHT_TILE_SIZE_Q6_K. Unpack k-group g of a tile to signed bytes (q - 32)
static inline HVX_Vector unpack_q6_k_group(const HVX_Vector * restrict vptr, int g, HVX_Vector mask_0f, HVX_Vector mask_03, HVX_Vector i32) {
    HVX_Vector v_lo = (g & 1) ? Q6_Vub_vlsr_VubR(vptr[g >> 1], 4) : Q6_V_vand_VV(vptr[g >> 1], mask_0f);
    HVX_Vector v_hi = (g & 3) ? Q6_Vub_vlsr_VubR(vptr[4 + (g >> 2)], 2 * (g & 3)) : vptr[4 + (g >> 2)];
    HVX_Vector v_q  = Q6_V_vor_VV(v_lo, Q6_Vw_vasl_VwR(Q6_V_vand_VV(v_hi, mask_03), 4));
    return Q6_Vb_vsub_VbVb(v_q, i32);
}

// k 0..15 and k 16..31 of a Q6_K tile have different scales: lo half of the pair sums k 0..15, hi half sums k 16..31
static inline HVX_VectorPair accum_q6_k_32x1(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act,
    HVX_Vector i32
) {
    HVX_Vector v_sum_lo = Q6_V_vzero();
    HVX_Vector v_sum_hi = Q6_V_vzero();
    HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q6_k_group(vptr, g,     mask_0f, mask_03, i32);
        HVX_Vector v_W_hi = unpack_q6_k_group(vptr, g + 4, mask_0f, mask_03, i32);
        v_sum_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum_lo, v_W_lo, v_act[g]);
        v_sum_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum_hi, v_W_hi, v_act[g + 4]);
    }

    return Q6_W_vcombine_VV(v_sum_hi, v_sum_lo);
}

static inline void accum_q6_k_32x2(
    const HVX_Vector * restrict vptr,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_Vector i32,
    HVX_VectorPair * v_sums0,
    HVX_VectorPair * v_sums1
) {
    HVX_Vector v_sum0_lo = Q6_V_vzero();
    HVX_Vector v_sum0_hi = Q6_V_vzero();
    HVX_Vector v_sum1_lo = Q6_V_vzero();
    HVX_Vector v_sum1_hi = Q6_V_vzero();
    HVX_Vector mask_0f = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector mask_03 = Q6_Vb_vsplat_R(0x03);

    #pragma unroll
    for (int g = 0; g < 4; g++) {
        HVX_Vector v_W_lo = unpack_q6_k_group(vptr, g,     mask_0f, mask_03, i32);
        HVX_Vector v_W_hi = unpack_q6_k_group(vptr, g + 4, mask_0f, mask_03, i32);
        v_sum0_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_lo, v_W_lo, v_act0[g]);
        v_sum0_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum0_hi, v_W_hi, v_act0[g + 4]);
        v_sum1_lo = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_lo, v_W_lo, v_act1[g]);
        v_sum1_hi = Q6_Vw_vrmpyacc_VwVbVb(v_sum1_hi, v_W_hi, v_act1[g + 4]);
    }

    *v_sums0 = Q6_W_vcombine_VV(v_sum0_hi, v_sum0_lo);
    *v_sums1 = Q6_W_vcombine_VV(v_sum1_hi, v_sum1_lo);
}

// scale the two half sums with the per-row tile scales (v_scale_w = vptr[6]) and the activation scale
static inline HVX_Vector scale_q6_k_32x1(HVX_VectorPair v_sums, HVX_Vector v_scale_w, HVX_Vector v_scale_a) {
    HVX_Vector v_scale_lo = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
    HVX_Vector v_scale_hi = hvx_vec_mul_f16_f16_to_f32_lower32(Q6_V_vror_VR(v_scale_w, 64), v_scale_a);
    HVX_Vector v_lo = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_lo_W(v_sums)), v_scale_lo);
    HVX_Vector v_hi = hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(Q6_V_hi_W(v_sums)), v_scale_hi);
    return hvx_vec_add_f32_f32(v_lo, v_hi);
}

// IQ2_S codebook in VTCM, copied by htp_iq2s_grid_ensure() (matmul-ops.c).
// grid/scratch are passed down from the op flag; NULL keeps the scalar
// .rodata lookup path.
#define IQ2S_GRID_BYTES 8192
#define IQ2S_LUT_BYTES  64

// IQ2_S direct dot helpers.
//
// Each IQ2_S grid entry contains eight positive int8 magnitudes.  Two adjacent
// vrmpy groups consume the low/high four bytes of the same grid entry, so
// decode both groups together.  This halves the codebook lookups and grid-index
// reconstruction work versus the original correctness-first implementation.
//
// Sign application is branchless and works on four packed bytes at once:
//   signed = (magnitude ^ neg_mask) + (neg_mask & 0x01010101)
// where each byte of neg_mask is either 0x00 or 0xff.  IQ2_S magnitudes are
// {8, 25, 43}, so the per-byte +1 cannot carry into an adjacent byte.
static const uint32_t iq2_s_sign4_lut[16] = {
    0x00000000u, 0x000000ffu, 0x0000ff00u, 0x0000ffffu,
    0x00ff0000u, 0x00ff00ffu, 0x00ffff00u, 0x00ffffffu,
    0xff000000u, 0xff0000ffu, 0xff00ff00u, 0xff00ffffu,
    0xffff0000u, 0xffff00ffu, 0xffffff00u, 0xffffffffu,
};

static inline uint32_t iq2_s_apply_sign4(uint32_t magnitude, uint32_t sign4) {
    const uint32_t neg = iq2_s_sign4_lut[sign4 & 0x0fu];
    return (magnitude ^ neg) + (neg & 0x01010101u);
}

// VTCM gather path: one 32-row word vector pair per group, built from the
// copied codebook with Q6_vgather instead of scalar .rodata loads.
static inline HVX_VectorPair iq2_s_unpack_group_8k_gather(const uint8_t * restrict tile, int l, const uint64_t * grid, HVX_Vector * restrict scratch) {
    const uint32_t grid_rt = (uint32_t) (uintptr_t) grid;
    const uint32_t lut_rt  = grid_rt + IQ2S_GRID_BYTES;
    // vgather Mu is the byte offset of the last valid byte, not the region size.
    const uint32_t grid_mu = IQ2S_GRID_BYTES - 1;
    const uint32_t lut_mu  = IQ2S_LUT_BYTES - 1;

    const HVX_Vector v_idx  = hvx_vmemu(tile + 0   + l * 32);
    const HVX_Vector v_sign = hvx_vmemu(tile + 128 + l * 32);
    const HVX_Vector v_qh   = hvx_vmemu(tile + 256);

    // grid byte offset = 8 * (idx + ((qh >> 2l) & 3) * 256)
    HVX_Vector v_off = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(
        Q6_V_vand_VV(Q6_Vub_vlsr_VubR(v_qh, 2 * l), Q6_Vb_vsplat_R(3))))));
    v_off = Q6_Vw_vasl_VwR(v_off, 8);                     // high index bits * 256
    HVX_Vector v_idx_w = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v_idx))));
    v_off = Q6_Vw_vadd_VwVw(v_off, v_idx_w);              // full grid index
    v_off = Q6_Vw_vadd_VwVw(v_off, v_off);                // *2
    v_off = Q6_Vw_vadd_VwVw(v_off, v_off);                // *4
    v_off = Q6_Vw_vadd_VwVw(v_off, v_off);                // *8 -> byte offset
    const HVX_Vector v_off_hi = Q6_Vw_vadd_VwVw(v_off, Q6_V_vsplat_R(4));

    // sign LUT word offset = 4 * nibble (low nibble feeds w0, high feeds w1)
    HVX_Vector v_off_s0 = Q6_Vw_vasl_VwR(
        Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(
            Q6_V_vand_VV(v_sign, Q6_Vb_vsplat_R(0x0f)))))), 2);
    HVX_Vector v_off_s1 = Q6_Vw_vasl_VwR(
        Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(
            Q6_Vub_vlsr_VubR(v_sign, 4))))), 2);

    // 4 gathers land in this thread's VTCM scratch slot (512 B per thread)
    Q6_vgather_ARMVw(&scratch[0], grid_rt, grid_mu, v_off);      // low 4 magnitudes
    Q6_vgather_ARMVw(&scratch[1], grid_rt, grid_mu, v_off_hi);   // high 4 magnitudes
    Q6_vgather_ARMVw(&scratch[2], lut_rt, lut_mu, v_off_s0);
    Q6_vgather_ARMVw(&scratch[3], lut_rt, lut_mu, v_off_s1);

    const HVX_Vector g0 = scratch[0];
    const HVX_Vector g1 = scratch[1];
    const HVX_Vector n0 = scratch[2];
    const HVX_Vector n1 = scratch[3];

    const HVX_Vector v_ones = Q6_V_vsplat_R(0x01010101);
    const HVX_Vector v_w0 = Q6_Vw_vadd_VwVw(Q6_V_vxor_VV(g0, n0), Q6_V_vand_VV(n0, v_ones));
    const HVX_Vector v_w1 = Q6_Vw_vadd_VwVw(Q6_V_vxor_VV(g1, n1), Q6_V_vand_VV(n1, v_ones));

    return Q6_W_vcombine_VV(v_w1, v_w0);
}

static inline HVX_VectorPair iq2_s_unpack_group_8k(const uint8_t * restrict tile, int l, const uint64_t * grid, HVX_Vector * restrict scratch) {
    // scratch is set from HTP_OPFLAGS_IQ2S_GATHER; NULL keeps the scalar path
    if (scratch) {
        return iq2_s_unpack_group_8k_gather(tile, l, grid, scratch);
    }

    uint32_t w0[32] __attribute__((aligned(128)));
    uint32_t w1[32] __attribute__((aligned(128)));

    const uint8_t * indices = tile + 0;
    const uint8_t * signs   = tile + 128;
    const uint8_t * qh      = tile + 256;

    const uint8_t * restrict idx_l  = indices + l * 32;
    const uint8_t * restrict sign_l = signs   + l * 32;
    const int qh_shift = 2 * l;

    #pragma unroll(4)
    for (int row = 0; row < 32; ++row) {
        const uint32_t grid_index =
            (uint32_t) idx_l[row] |
            (((uint32_t) qh[row] >> qh_shift & 0x03u) << 8);

        const uint64_t grid = iq2s_grid[grid_index];
        const uint8_t sign_mask = sign_l[row];

        w0[row] = iq2_s_apply_sign4((uint32_t) grid,        sign_mask);
        w1[row] = iq2_s_apply_sign4((uint32_t) (grid >> 32), sign_mask >> 4);
    }

    return Q6_W_vcombine_VV(*(const HVX_Vector *) w1, *(const HVX_Vector *) w0);
}

static inline HVX_Vector iq2_s_scale_vector(const uint8_t * restrict tile) {
    // scales[] packs two 4-bit multipliers per row.  Build the 64 fp16 weight
    // scales entirely in HVX:
    //   scale = d * (nibble + 0.5) * 0.25
    //
    // Only the first 32 bytes of v_sc_raw are scales.  After widening bytes
    // to halfwords, those 32 values occupy the low 64 bytes of each vector.
    // Put low nibbles in the first 32 fp16 lanes and high nibbles in the
    // second 32 lanes, matching scale_q6_k_32x1().
    const HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0f);
    const HVX_Vector v_sc_raw = hvx_vmemu(tile + 288);

    const HVX_Vector v_sc_lo_b = Q6_V_vand_VV(v_sc_raw, mask_h4);
    const HVX_Vector v_sc_hi_b = Q6_Vub_vlsr_VubR(v_sc_raw, 4);

    const HVX_Vector v_sc_lo_h = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v_sc_lo_b));
    const HVX_Vector v_sc_hi_h = Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v_sc_hi_b));

    const HVX_VectorPred q_low64 = Q6_Q_vsetq2_R(64);
    const HVX_Vector v_nibble_h = Q6_V_vmux_QVV(
        q_low64,
        v_sc_lo_h,
        Q6_V_vror_VR(v_sc_hi_h, 64));

    const HVX_Vector v_nibble_hf = Q6_Vhf_equals_Vh(v_nibble_h);
    const HVX_Vector v_half      = hvx_vec_splat_f16((__fp16) 0.5f);
    const HVX_Vector v_quarter   = hvx_vec_splat_f16((__fp16) 0.25f);
    const HVX_Vector v_factor = hvx_vec_mul_f16_f16(
        hvx_vec_add_f16_f16(v_nibble_hf, v_half),
        v_quarter);

    // d[] contains 32 fp16 values (64 bytes).  Duplicate that lower half into
    // both 32-lane halves so it lines up with the low/high scale nibbles.
    const HVX_Vector v_d_raw = hvx_vmemu(tile + 320);
    const HVX_Vector v_d = Q6_V_vmux_QVV(
        q_low64,
        v_d_raw,
        Q6_V_vror_VR(v_d_raw, 64));

    return hvx_vec_mul_f16_f16(v_d, v_factor);
}

static inline HVX_VectorPair accum_iq2_s_32x1(
    const uint8_t * restrict tile,
    const HVX_Vector * restrict v_act,
    const uint64_t * grid,
    HVX_Vector * restrict scratch
) {
    HVX_Vector lo = Q6_V_vzero();
    HVX_Vector hi = Q6_V_vzero();

    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const HVX_VectorPair v_w = iq2_s_unpack_group_8k(tile, l, grid, scratch);
        if (l < 2) {
            lo = Q6_Vw_vrmpyacc_VwVbVb(lo, Q6_V_lo_W(v_w), v_act[2 * l + 0]);
            lo = Q6_Vw_vrmpyacc_VwVbVb(lo, Q6_V_hi_W(v_w), v_act[2 * l + 1]);
        } else {
            hi = Q6_Vw_vrmpyacc_VwVbVb(hi, Q6_V_lo_W(v_w), v_act[2 * l + 0]);
            hi = Q6_Vw_vrmpyacc_VwVbVb(hi, Q6_V_hi_W(v_w), v_act[2 * l + 1]);
        }
    }

    return Q6_W_vcombine_VV(hi, lo);
}

static inline void accum_iq2_s_32x2(
    const uint8_t * restrict tile,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_VectorPair * restrict sums0,
    HVX_VectorPair * restrict sums1,
    const uint64_t * grid,
    HVX_Vector * restrict scratch
) {
    HVX_Vector lo0 = Q6_V_vzero();
    HVX_Vector hi0 = Q6_V_vzero();
    HVX_Vector lo1 = Q6_V_vzero();
    HVX_Vector hi1 = Q6_V_vzero();

    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const HVX_VectorPair v_w = iq2_s_unpack_group_8k(tile, l, grid, scratch);
        const HVX_Vector w0 = Q6_V_lo_W(v_w);
        const HVX_Vector w1 = Q6_V_hi_W(v_w);

        if (l < 2) {
            lo0 = Q6_Vw_vrmpyacc_VwVbVb(lo0, w0, v_act0[2 * l + 0]);
            lo0 = Q6_Vw_vrmpyacc_VwVbVb(lo0, w1, v_act0[2 * l + 1]);
            lo1 = Q6_Vw_vrmpyacc_VwVbVb(lo1, w0, v_act1[2 * l + 0]);
            lo1 = Q6_Vw_vrmpyacc_VwVbVb(lo1, w1, v_act1[2 * l + 1]);
        } else {
            hi0 = Q6_Vw_vrmpyacc_VwVbVb(hi0, w0, v_act0[2 * l + 0]);
            hi0 = Q6_Vw_vrmpyacc_VwVbVb(hi0, w1, v_act0[2 * l + 1]);
            hi1 = Q6_Vw_vrmpyacc_VwVbVb(hi1, w0, v_act1[2 * l + 0]);
            hi1 = Q6_Vw_vrmpyacc_VwVbVb(hi1, w1, v_act1[2 * l + 1]);
        }
    }

    *sums0 = Q6_W_vcombine_VV(hi0, lo0);
    *sums1 = Q6_W_vcombine_VV(hi1, lo1);
}

static void tiled_vec_dot_iq2_s_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz, const uint64_t * grid, HVX_Vector * restrict scratch) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    const uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
        const uint8_t * restrict tile = tile_ptr + kt * 384;
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        const HVX_VectorPair sums = accum_iq2_s_32x1(tile, v_act, grid, scratch);
        const HVX_Vector v_scale_w = iq2_s_scale_vector(tile);
        v_sum_float = hvx_vec_add_f32_f32(
            v_sum_float,
            scale_q6_k_32x1(sums, v_scale_w, v_act[8]));
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_iq2_s_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1, const uint64_t * grid, HVX_Vector * restrict scratch) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector sum0 = Q6_V_vzero();
    HVX_Vector sum1 = Q6_V_vzero();

    const uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
        const uint8_t * restrict tile = tile_ptr + kt * 384;
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair sums0;
        HVX_VectorPair sums1;
        accum_iq2_s_32x2(tile, v_act0, v_act1, &sums0, &sums1, grid, scratch);

        const HVX_Vector v_scale_w = iq2_s_scale_vector(tile);
        sum0 = hvx_vec_add_f32_f32(sum0, scale_q6_k_32x1(sums0, v_scale_w, v_act0[8]));
        sum1 = hvx_vec_add_f32_f32(sum1, scale_q6_k_32x1(sums1, v_scale_w, v_act1[8]));
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(sum0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), sum0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(sum1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), sum1);
    }
}

// IQ2_XS / IQ2_XXS share the IQ2_S tile shape (see iq2-family-repack.h): the
// sign plane carries the sign index directly instead of a qh field, so one
// unpack serves both types.  Only the grid table and its VTCM size differ and
// are passed in.
#define IQ2F_XS_GRID_BYTES  4096
#define IQ2F_XXS_GRID_BYTES 2048
#define IQ2F_SIGNS_BYTES    1024

static inline uint32_t iq2f_apply_neg4(uint32_t magnitude, uint32_t neg) {
    return (magnitude ^ neg) + (neg & 0x01010101u);
}

// VTCM gather path: two grid words and two sign words per group, landing in
// the same per-thread scratch slot the IQ2_S gather uses.
static inline HVX_VectorPair iq2f_unpack_group_8k_gather(
    const uint8_t * restrict tile, int l,
    const uint64_t * grid, uint32_t grid_bytes,
    const uint64_t * signs,
    HVX_Vector * restrict scratch) {

    const uint32_t grid_rt  = (uint32_t) (uintptr_t) grid;
    const uint32_t signs_rt = (uint32_t) (uintptr_t) signs;
    const uint32_t grid_mu  = grid_bytes - 1;
    const uint32_t signs_mu = IQ2F_SIGNS_BYTES - 1;

    const HVX_Vector v_idx  = hvx_vmemu(tile + 0   + l * 32);
    const HVX_Vector v_sign = hvx_vmemu(tile + 128 + l * 32);

    // grid byte offset = 8 * (idx | ((sign & 1) << 8))
    HVX_Vector v_off = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(v_idx))));
    const HVX_Vector v_hi = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(
        Q6_V_vand_VV(v_sign, Q6_Vb_vsplat_R(1))))));
    v_off = Q6_Vw_vadd_VwVw(v_off, Q6_Vw_vasl_VwR(v_hi, 8));
    v_off = Q6_Vw_vadd_VwVw(v_off, v_off);
    v_off = Q6_Vw_vadd_VwVw(v_off, v_off);
    v_off = Q6_Vw_vadd_VwVw(v_off, v_off);

    // sign byte offset = 8 * (sign >> 1)
    const HVX_Vector v_sidx = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(
        Q6_Vub_vlsr_VubR(v_sign, 1)))));
    const HVX_Vector v_off_s = Q6_Vw_vasl_VwR(v_sidx, 3);

    Q6_vgather_ARMVw(&scratch[0], grid_rt,  grid_mu,  v_off);
    Q6_vgather_ARMVw(&scratch[1], grid_rt,  grid_mu,  Q6_Vw_vadd_VwVw(v_off, Q6_V_vsplat_R(4)));
    Q6_vgather_ARMVw(&scratch[2], signs_rt, signs_mu, v_off_s);
    Q6_vgather_ARMVw(&scratch[3], signs_rt, signs_mu, Q6_Vw_vadd_VwVw(v_off_s, Q6_V_vsplat_R(4)));

    const HVX_Vector v_ones = Q6_V_vsplat_R(0x01010101);
    const HVX_Vector v_w0 = Q6_Vw_vadd_VwVw(
        Q6_V_vxor_VV(scratch[0], scratch[2]), Q6_V_vand_VV(scratch[2], v_ones));
    const HVX_Vector v_w1 = Q6_Vw_vadd_VwVw(
        Q6_V_vxor_VV(scratch[1], scratch[3]), Q6_V_vand_VV(scratch[3], v_ones));

    return Q6_W_vcombine_VV(v_w1, v_w0);
}

// grid/signs point at .rodata when the gather flag is off and at VTCM when it
// is on; a NULL scratch keeps the scalar lookup path.
static inline HVX_VectorPair iq2f_unpack_group_8k(
    const uint8_t * restrict tile, int l,
    const uint64_t * grid, uint32_t grid_bytes,
    const uint64_t * signs,
    HVX_Vector * restrict scratch) {

    if (scratch) {
        return iq2f_unpack_group_8k_gather(tile, l, grid, grid_bytes, signs, scratch);
    }

    uint32_t w0[32] __attribute__((aligned(128)));
    uint32_t w1[32] __attribute__((aligned(128)));

    const uint8_t * restrict idx_l  = tile + 0   + l * 32;
    const uint8_t * restrict sign_l = tile + 128 + l * 32;

    #pragma unroll(4)
    for (int row = 0; row < 32; ++row) {
        const uint32_t payload = sign_l[row];
        const uint32_t grid_index = (uint32_t) idx_l[row] | ((payload & 1u) << 8);
        const uint64_t g = grid[grid_index];
        const uint64_t s = signs[payload >> 1];

        w0[row] = iq2f_apply_neg4((uint32_t) g,         (uint32_t) s);
        w1[row] = iq2f_apply_neg4((uint32_t) (g >> 32), (uint32_t) (s >> 32));
    }

    return Q6_W_vcombine_VV(*(const HVX_Vector *) w1, *(const HVX_Vector *) w0);
}

static inline HVX_VectorPair accum_iq2f_32x1(
    const uint8_t * restrict tile,
    const HVX_Vector * restrict v_act,
    const uint64_t * grid, uint32_t grid_bytes,
    const uint64_t * signs,
    HVX_Vector * restrict scratch
) {
    HVX_Vector lo = Q6_V_vzero();
    HVX_Vector hi = Q6_V_vzero();

    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const HVX_VectorPair v_w = iq2f_unpack_group_8k(tile, l, grid, grid_bytes, signs, scratch);
        if (l < 2) {
            lo = Q6_Vw_vrmpyacc_VwVbVb(lo, Q6_V_lo_W(v_w), v_act[2 * l + 0]);
            lo = Q6_Vw_vrmpyacc_VwVbVb(lo, Q6_V_hi_W(v_w), v_act[2 * l + 1]);
        } else {
            hi = Q6_Vw_vrmpyacc_VwVbVb(hi, Q6_V_lo_W(v_w), v_act[2 * l + 0]);
            hi = Q6_Vw_vrmpyacc_VwVbVb(hi, Q6_V_hi_W(v_w), v_act[2 * l + 1]);
        }
    }

    return Q6_W_vcombine_VV(hi, lo);
}

static inline void accum_iq2f_32x2(
    const uint8_t * restrict tile,
    const HVX_Vector * restrict v_act0,
    const HVX_Vector * restrict v_act1,
    HVX_VectorPair * restrict sums0,
    HVX_VectorPair * restrict sums1,
    const uint64_t * grid, uint32_t grid_bytes,
    const uint64_t * signs,
    HVX_Vector * restrict scratch
) {
    HVX_Vector lo0 = Q6_V_vzero();
    HVX_Vector hi0 = Q6_V_vzero();
    HVX_Vector lo1 = Q6_V_vzero();
    HVX_Vector hi1 = Q6_V_vzero();

    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const HVX_VectorPair v_w = iq2f_unpack_group_8k(tile, l, grid, grid_bytes, signs, scratch);
        const HVX_Vector w0 = Q6_V_lo_W(v_w);
        const HVX_Vector w1 = Q6_V_hi_W(v_w);

        if (l < 2) {
            lo0 = Q6_Vw_vrmpyacc_VwVbVb(lo0, w0, v_act0[2 * l + 0]);
            lo0 = Q6_Vw_vrmpyacc_VwVbVb(lo0, w1, v_act0[2 * l + 1]);
            lo1 = Q6_Vw_vrmpyacc_VwVbVb(lo1, w0, v_act1[2 * l + 0]);
            lo1 = Q6_Vw_vrmpyacc_VwVbVb(lo1, w1, v_act1[2 * l + 1]);
        } else {
            hi0 = Q6_Vw_vrmpyacc_VwVbVb(hi0, w0, v_act0[2 * l + 0]);
            hi0 = Q6_Vw_vrmpyacc_VwVbVb(hi0, w1, v_act0[2 * l + 1]);
            hi1 = Q6_Vw_vrmpyacc_VwVbVb(hi1, w0, v_act1[2 * l + 0]);
            hi1 = Q6_Vw_vrmpyacc_VwVbVb(hi1, w1, v_act1[2 * l + 1]);
        }
    }

    *sums0 = Q6_W_vcombine_VV(hi0, lo0);
    *sums1 = Q6_W_vcombine_VV(hi1, lo1);
}

static void tiled_vec_dot_iq2f_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz, const uint64_t * grid, uint32_t grid_bytes, const uint64_t * signs, HVX_Vector * restrict scratch) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    const uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
        const uint8_t * restrict tile = tile_ptr + kt * 384;
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        const HVX_VectorPair sums = accum_iq2f_32x1(tile, v_act, grid, grid_bytes, signs, scratch);
        const HVX_Vector v_scale_w = iq2_s_scale_vector(tile);
        v_sum_float = hvx_vec_add_f32_f32(
            v_sum_float,
            scale_q6_k_32x1(sums, v_scale_w, v_act[8]));
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_iq2f_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1, const uint64_t * grid, uint32_t grid_bytes, const uint64_t * signs, HVX_Vector * restrict scratch) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector sum0 = Q6_V_vzero();
    HVX_Vector sum1 = Q6_V_vzero();

    const uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; ++kt) {
        const uint8_t * restrict tile = tile_ptr + kt * 384;
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair sums0;
        HVX_VectorPair sums1;
        accum_iq2f_32x2(tile, v_act0, v_act1, &sums0, &sums1, grid, grid_bytes, signs, scratch);

        const HVX_Vector v_scale_w = iq2_s_scale_vector(tile);
        sum0 = hvx_vec_add_f32_f32(sum0, scale_q6_k_32x1(sums0, v_scale_w, v_act0[8]));
        sum1 = hvx_vec_add_f32_f32(sum1, scale_q6_k_32x1(sums1, v_scale_w, v_act1[8]));
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(sum0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), sum0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(sum1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), sum1);
    }
}

// IQ3_XXS shares the Q6_K activation tile (see iq3-xxs-repack.h).  The scalar
// path reads the tile planes directly and is only selected with the gather
// op flag off; the gather path lives in tiled_vec_dot_iq3_xxs_gather_32x*.
static void tiled_vec_dot_iq3_xxs_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * weights = vx;
    const uint8_t * activation = vy;
    for (uint32_t row = 0; row < valid_rows; ++row) {
        float sum = 0.0f;
        for (uint32_t kt = 0; kt < n / 32; ++kt) {
            const uint8_t * tile = weights + kt * HTP_MM_WEIGHT_TILE_SIZE_IQ3_XXS;
            const int8_t * act = (const int8_t *) (activation + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
            uint32_t aux;
            __fp16 d, da;
            // Hexagon reads little-endian aux words, as the GGUF CPU reference does.
            memcpy(&aux, tile + 256 + 4 * row, sizeof(aux));
            memcpy(&d, tile + 384 + 2 * row, sizeof(d));
            memcpy(&da, act + 1024, sizeof(da));
            const float db = (float) d * (0.5f + (aux >> 28)) * 0.5f;
            int32_t dot = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) {
                    const uint32_t grid = iq3xxs_grid[tile[(2 * l + j / 4) * 32 + row]];
                    const int magnitude = (grid >> (8 * (j % 4))) & 255;
                    const int weight = (signs & (1u << j)) ? -magnitude : magnitude;
                    dot += weight * act[(2 * l + j / 4) * 128 + j % 4];
                }
            }
            sum += (float) dot * db * (float) da;
        }
        s[row] = sum + (sz ? sz[row] : 0.0f);
    }
}

static void tiled_vec_dot_iq3_xxs_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    tiled_vec_dot_iq3_xxs_32x1(n, s0, vx, vy0, valid_rows, sz0);
    tiled_vec_dot_iq3_xxs_32x1(n, s1, vx, vy1, valid_rows, sz1);
}

static void tiled_vec_dot_iq3_s_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * weights = vx;
    const uint8_t * activation = vy;
    for (uint32_t row = 0; row < valid_rows; ++row) {
        float sum = 0.0f;
        for (uint32_t kt = 0; kt < n / 32; ++kt) {
            const uint8_t * tile = weights + kt * HTP_MM_WEIGHT_TILE_SIZE_IQ3_S;
            const int8_t * act = (const int8_t *) (activation + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
            const uint8_t qh = tile[HTP_IQ3S_QH_PLANE_OFFSET + row];
            const uint8_t scale = tile[HTP_IQ3S_SCALE_PLANE_OFFSET + row];
            __fp16 d, da;
            memcpy(&d, tile + HTP_IQ3S_D_PLANE_OFFSET + 2 * row, sizeof(d));
            memcpy(&da, act + 1024, sizeof(da));
            const float db = (float) d * (1 + 2 * scale);
            int32_t dot = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = tile[HTP_IQ3S_SIGN_PLANE_OFFSET + 32 * l + row];
                for (int j = 0; j < 8; ++j) {
                    const int i = 2 * l + j / 4;
                    const unsigned idx = tile[HTP_IQ3S_INDEX_PLANE_OFFSET + 32 * i + row] | (((qh >> i) & 1u) << 8);
                    const int magnitude = (iq3s_grid[idx] >> (8 * (j % 4))) & 255;
                    const int weight = (signs & (1u << j)) ? -magnitude : magnitude;
                    dot += weight * act[i * 128 + j % 4];
                }
            }
            sum += (float) dot * db * (float) da;
        }
        s[row] = sum + (sz ? sz[row] : 0.0f);
    }
}

static void tiled_vec_dot_iq3_s_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    tiled_vec_dot_iq3_s_32x1(n, s0, vx, vy0, valid_rows, sz0);
    tiled_vec_dot_iq3_s_32x1(n, s1, vx, vy1, valid_rows, sz1);
}

static inline HVX_VectorPair iq3_xxs_unpack_group_gather(const uint8_t * tile, HVX_Vector aux, int l, const uint32_t * grid, const uint8_t * signs, HVX_Vector * restrict scratch) {
    HVX_Vector idx0 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmemu(tile + 64 * l)))));
    HVX_Vector idx1 = Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmemu(tile + 64 * l + 32)))));
    HVX_Vector sign_index = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(aux, 7 * l), Q6_V_vsplat_R(127));
    HVX_Vector sign_offset = Q6_V_vand_VV(sign_index, Q6_V_vsplat_R(124));

    // Each destination is a 128-byte slot in this thread's VTCM scratch.
    Q6_vgather_ARMVw(&scratch[0], (uint32_t) (uintptr_t) grid, 1023, Q6_Vw_vasl_VwR(idx0, 2));
    Q6_vgather_ARMVw(&scratch[1], (uint32_t) (uintptr_t) grid, 1023, Q6_Vw_vasl_VwR(idx1, 2));
    Q6_vgather_ARMVw(&scratch[2], (uint32_t) (uintptr_t) signs, 127, sign_offset);

    // Select one byte from each aligned sign-table word, then repeat it four times.
    HVX_Vector shift = Q6_Vw_vasl_VwR(Q6_V_vand_VV(sign_index, Q6_V_vsplat_R(3)), 3);
    HVX_Vector sign = Q6_V_vand_VV(Q6_Vw_vlsr_VwVw(scratch[2], shift), Q6_V_vsplat_R(255));
    sign = Q6_V_vor_VV(sign, Q6_Vw_vasl_VwR(sign, 8));
    sign = Q6_V_vor_VV(sign, Q6_Vw_vasl_VwR(sign, 16));
    HVX_VectorPred neg0 = Q6_Q_vcmp_gt_VubVub(Q6_V_vand_VV(sign, Q6_V_vsplat_R(0x08040201)), Q6_V_vzero());
    HVX_VectorPred neg1 = Q6_Q_vcmp_gt_VubVub(Q6_V_vand_VV(sign, Q6_V_vsplat_R(0x80402010)), Q6_V_vzero());
    HVX_Vector g0 = scratch[0];
    HVX_Vector g1 = scratch[1];
    HVX_Vector w0 = Q6_V_vmux_QVV(neg0, Q6_Vb_vsub_VbVb(Q6_V_vzero(), g0), g0);
    HVX_Vector w1 = Q6_V_vmux_QVV(neg1, Q6_Vb_vsub_VbVb(Q6_V_vzero(), g1), g1);
    return Q6_W_vcombine_VV(w1, w0);
}

static inline HVX_Vector iq3_xxs_scale_vector(const uint8_t * tile, HVX_Vector aux, HVX_Vector activation_scale) {
    HVX_Vector nibble = Q6_Vsf_equals_Vw(Q6_Vuw_vlsr_VuwR(aux, 28));
    HVX_Vector half = hvx_vec_splat_f32(0.5f);
    HVX_Vector factor = hvx_vec_mul_f32_f32(hvx_vec_add_f32_f32(nibble, half), half);
    // Keep the fp16 d and activation scales; compute their product in fp32.
    HVX_Vector base = hvx_vec_mul_f16_f16_to_f32_lower32(hvx_vmemu(tile + 384), activation_scale);
    return hvx_vec_mul_f32_f32(base, factor);
}

static void tiled_vec_dot_iq3_xxs_gather_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz, const uint32_t * grid, const uint8_t * signs, HVX_Vector * restrict scratch) {
    if (!scratch) {
        tiled_vec_dot_iq3_xxs_32x1(n, s, vx, vy, valid_rows, sz);
        return;
    }
    const uint8_t * weights = vx;
    const uint8_t * activation = vy;
    HVX_Vector sum = Q6_V_vzero();
    for (uint32_t kt = 0; kt < n / 32; ++kt) {
        const uint8_t * tile = weights + kt * HTP_MM_WEIGHT_TILE_SIZE_IQ3_XXS;
        const HVX_Vector * act = (const HVX_Vector *) (activation + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
        HVX_Vector aux = hvx_vmemu(tile + 256);
        HVX_Vector dot = Q6_V_vzero();
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            HVX_VectorPair w = iq3_xxs_unpack_group_gather(tile, aux, l, grid, signs, scratch);
            dot = Q6_Vw_vrmpyacc_VwVbVb(dot, Q6_V_lo_W(w), act[2 * l]);
            dot = Q6_Vw_vrmpyacc_VwVbVb(dot, Q6_V_hi_W(w), act[2 * l + 1]);
        }
        sum = hvx_vec_add_f32_f32(sum, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(dot), iq3_xxs_scale_vector(tile, aux, act[8])));
    }
    if (sz) {
        sum = hvx_vec_add_f32_f32(sum, hvx_vmemu(sz));
    }
    hvx_vec_store_u(s, valid_rows * sizeof(float), sum);
}

static void tiled_vec_dot_iq3_xxs_gather_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1, const uint32_t * grid, const uint8_t * signs, HVX_Vector * restrict scratch) {
    if (!scratch) {
        tiled_vec_dot_iq3_xxs_32x2(n, s0, s1, vx, vy0, vy1, valid_rows, sz0, sz1);
        return;
    }
    const uint8_t * weights = vx;
    const uint8_t * activation0 = vy0;
    const uint8_t * activation1 = vy1;
    HVX_Vector sum0 = Q6_V_vzero();
    HVX_Vector sum1 = Q6_V_vzero();
    for (uint32_t kt = 0; kt < n / 32; ++kt) {
        const uint8_t * tile = weights + kt * HTP_MM_WEIGHT_TILE_SIZE_IQ3_XXS;
        const HVX_Vector * act0 = (const HVX_Vector *) (activation0 + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
        const HVX_Vector * act1 = (const HVX_Vector *) (activation1 + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
        HVX_Vector aux = hvx_vmemu(tile + 256);
        HVX_Vector dot0 = Q6_V_vzero();
        HVX_Vector dot1 = Q6_V_vzero();
        #pragma unroll(1)
        for (int l = 0; l < 4; ++l) {
            HVX_VectorPair w = iq3_xxs_unpack_group_gather(tile, aux, l, grid, signs, scratch);
            dot0 = Q6_Vw_vrmpyacc_VwVbVb(dot0, Q6_V_lo_W(w), act0[2 * l]);
            dot0 = Q6_Vw_vrmpyacc_VwVbVb(dot0, Q6_V_hi_W(w), act0[2 * l + 1]);
            dot1 = Q6_Vw_vrmpyacc_VwVbVb(dot1, Q6_V_lo_W(w), act1[2 * l]);
            dot1 = Q6_Vw_vrmpyacc_VwVbVb(dot1, Q6_V_hi_W(w), act1[2 * l + 1]);
        }
        sum0 = hvx_vec_add_f32_f32(sum0, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(dot0), iq3_xxs_scale_vector(tile, aux, act0[8])));
        sum1 = hvx_vec_add_f32_f32(sum1, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(dot1), iq3_xxs_scale_vector(tile, aux, act1[8])));
    }
    if (sz0) {
        sum0 = hvx_vec_add_f32_f32(sum0, hvx_vmemu(sz0));
    }
    if (sz1) {
        sum1 = hvx_vec_add_f32_f32(sum1, hvx_vmemu(sz1));
    }
    hvx_vec_store_u(s0, valid_rows * sizeof(float), sum0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), sum1);
}

static inline HVX_Vector iq3_s_widen_vector(HVX_Vector bytes) {
    return Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(bytes))));
}

static inline HVX_Vector iq3_s_widen_bytes(const uint8_t * p) {
    return iq3_s_widen_vector(hvx_vmemu(p));
}

static inline HVX_VectorPair iq3_s_unpack_group_gather(const uint8_t * tile, HVX_Vector qh, int l, const uint32_t * grid, HVX_Vector * restrict scratch) {
    HVX_Vector idx0 = iq3_s_widen_bytes(tile + HTP_IQ3S_INDEX_PLANE_OFFSET + 64 * l);
    HVX_Vector idx1 = iq3_s_widen_bytes(tile + HTP_IQ3S_INDEX_PLANE_OFFSET + 64 * l + 32);
    HVX_Vector bit0 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(qh, 2 * l), Q6_V_vsplat_R(1));
    HVX_Vector bit1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(qh, 2 * l + 1), Q6_V_vsplat_R(1));
    idx0 = Q6_V_vor_VV(idx0, Q6_Vw_vasl_VwR(bit0, 8));
    idx1 = Q6_V_vor_VV(idx1, Q6_Vw_vasl_VwR(bit1, 8));

    // Both destinations are this thread's VTCM scratch; Mu is the last valid byte offset.
    Q6_vgather_ARMVw(&scratch[0], (uint32_t) (uintptr_t) grid, 2047, Q6_Vw_vasl_VwR(idx0, 2));
    Q6_vgather_ARMVw(&scratch[1], (uint32_t) (uintptr_t) grid, 2047, Q6_Vw_vasl_VwR(idx1, 2));

    HVX_Vector sign = iq3_s_widen_bytes(tile + HTP_IQ3S_SIGN_PLANE_OFFSET + 32 * l);
    sign = Q6_V_vor_VV(sign, Q6_Vw_vasl_VwR(sign, 8));
    sign = Q6_V_vor_VV(sign, Q6_Vw_vasl_VwR(sign, 16));
    HVX_VectorPred neg0 = Q6_Q_vcmp_gt_VubVub(Q6_V_vand_VV(sign, Q6_V_vsplat_R(0x08040201)), Q6_V_vzero());
    HVX_VectorPred neg1 = Q6_Q_vcmp_gt_VubVub(Q6_V_vand_VV(sign, Q6_V_vsplat_R(0x80402010)), Q6_V_vzero());
    HVX_Vector g0 = scratch[0];
    HVX_Vector g1 = scratch[1];
    HVX_Vector w0 = Q6_V_vmux_QVV(neg0, Q6_Vb_vsub_VbVb(Q6_V_vzero(), g0), g0);
    HVX_Vector w1 = Q6_V_vmux_QVV(neg1, Q6_Vb_vsub_VbVb(Q6_V_vzero(), g1), g1);
    return Q6_W_vcombine_VV(w1, w0);
}

static inline HVX_Vector iq3_s_scale_vector(const uint8_t * tile, HVX_Vector activation_scale) {
    HVX_Vector tail = hvx_vmemu(tile + HTP_IQ3S_QH_PLANE_OFFSET);
    HVX_Vector nibble = iq3_s_widen_vector(Q6_V_vror_VR(tail, HTP_IQ3S_SCALE_PLANE_OFFSET - HTP_IQ3S_QH_PLANE_OFFSET));
    HVX_Vector factor = Q6_Vsf_equals_Vw(Q6_Vw_vadd_VwVw(Q6_Vw_vasl_VwR(nibble, 1), Q6_V_vsplat_R(1)));
    HVX_Vector d = Q6_V_vror_VR(tail, HTP_IQ3S_D_PLANE_OFFSET - HTP_IQ3S_QH_PLANE_OFFSET);
    HVX_Vector base = hvx_vec_mul_f16_f16_to_f32_lower32(d, activation_scale);
    return hvx_vec_mul_f32_f32(base, factor);
}

static void tiled_vec_dot_iq3_s_gather_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz, const uint32_t * grid, HVX_Vector * restrict scratch) {
    if (!scratch) {
        tiled_vec_dot_iq3_s_32x1(n, s, vx, vy, valid_rows, sz);
        return;
    }
    const uint8_t * weights = vx;
    const uint8_t * activation = vy;
    HVX_Vector sum = Q6_V_vzero();
    for (uint32_t kt = 0; kt < n / 32; ++kt) {
        const uint8_t * tile = weights + kt * HTP_MM_WEIGHT_TILE_SIZE_IQ3_S;
        const HVX_Vector * act = (const HVX_Vector *) (activation + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
        HVX_Vector qh = iq3_s_widen_bytes(tile + HTP_IQ3S_QH_PLANE_OFFSET);
        HVX_Vector dot = Q6_V_vzero();
        #pragma unroll(1)
        for (int l = 0; l < 4; ++l) {
            HVX_VectorPair w = iq3_s_unpack_group_gather(tile, qh, l, grid, scratch);
            dot = Q6_Vw_vrmpyacc_VwVbVb(dot, Q6_V_lo_W(w), act[2 * l]);
            dot = Q6_Vw_vrmpyacc_VwVbVb(dot, Q6_V_hi_W(w), act[2 * l + 1]);
        }
        sum = hvx_vec_add_f32_f32(sum, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(dot), iq3_s_scale_vector(tile, act[8])));
    }
    if (sz) {
        sum = hvx_vec_add_f32_f32(sum, hvx_vmemu(sz));
    }
    hvx_vec_store_u(s, valid_rows * sizeof(float), sum);
}

static void tiled_vec_dot_iq3_s_gather_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1, const uint32_t * grid, HVX_Vector * restrict scratch) {
    if (!scratch) {
        tiled_vec_dot_iq3_s_32x2(n, s0, s1, vx, vy0, vy1, valid_rows, sz0, sz1);
        return;
    }
    const uint8_t * weights = vx;
    const uint8_t * activation0 = vy0;
    const uint8_t * activation1 = vy1;
    HVX_Vector sum0 = Q6_V_vzero();
    HVX_Vector sum1 = Q6_V_vzero();
    for (uint32_t kt = 0; kt < n / 32; ++kt) {
        const uint8_t * tile = weights + kt * HTP_MM_WEIGHT_TILE_SIZE_IQ3_S;
        const HVX_Vector * act0 = (const HVX_Vector *) (activation0 + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
        const HVX_Vector * act1 = (const HVX_Vector *) (activation1 + kt * HTP_MM_ACT_TILE_SIZE_Q8_0);
        HVX_Vector qh = iq3_s_widen_bytes(tile + HTP_IQ3S_QH_PLANE_OFFSET);
        HVX_Vector dot0 = Q6_V_vzero();
        HVX_Vector dot1 = Q6_V_vzero();
        #pragma unroll(1)
        for (int l = 0; l < 4; ++l) {
            HVX_VectorPair w = iq3_s_unpack_group_gather(tile, qh, l, grid, scratch);
            dot0 = Q6_Vw_vrmpyacc_VwVbVb(dot0, Q6_V_lo_W(w), act0[2 * l]);
            dot0 = Q6_Vw_vrmpyacc_VwVbVb(dot0, Q6_V_hi_W(w), act0[2 * l + 1]);
            dot1 = Q6_Vw_vrmpyacc_VwVbVb(dot1, Q6_V_lo_W(w), act1[2 * l]);
            dot1 = Q6_Vw_vrmpyacc_VwVbVb(dot1, Q6_V_hi_W(w), act1[2 * l + 1]);
        }
        sum0 = hvx_vec_add_f32_f32(sum0, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(dot0), iq3_s_scale_vector(tile, act0[8])));
        sum1 = hvx_vec_add_f32_f32(sum1, hvx_vec_mul_f32_f32(Q6_Vsf_equals_Vw(dot1), iq3_s_scale_vector(tile, act1[8])));
    }
    if (sz0) {
        sum0 = hvx_vec_add_f32_f32(sum0, hvx_vmemu(sz0));
    }
    if (sz1) {
        sum1 = hvx_vec_add_f32_f32(sum1, hvx_vmemu(sz1));
    }
    hvx_vec_store_u(s0, valid_rows * sizeof(float), sum0);
    hvx_vec_store_u(s1, valid_rows * sizeof(float), sum1);
}

static inline HVX_Vector q2k_widen_bytes_32(const uint8_t * src) {
    return Q6_V_lo_W(Q6_Ww_vunpack_Vh(Q6_V_lo_W(Q6_Wuh_vunpack_Vub(hvx_vmemu(src)))));
}

static inline HVX_Vector q2k_unpack_quant_plane_32(const uint8_t * tile, int plane) {
    HVX_Vector packed = q2k_widen_bytes_32(tile + Q2K_QUANT_PLANE_OFFSET + 32 * plane);
    HVX_Vector mask = Q6_V_vsplat_R(3);
    HVX_Vector q0 = Q6_V_vand_VV(packed, mask);
    HVX_Vector q1 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(packed, 2), mask);
    HVX_Vector q2 = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(packed, 4), mask);
    HVX_Vector q3 = Q6_Vuw_vlsr_VuwR(packed, 6);
    return Q6_V_vor_VV(Q6_V_vor_VV(q0, Q6_Vw_vasl_VwR(q1, 8)),
                       Q6_V_vor_VV(Q6_Vw_vasl_VwR(q2, 16), Q6_Vw_vasl_VwR(q3, 24)));
}

static void tiled_vec_dot_q4_0_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, i8);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q4_0_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector i8 = Q6_Vb_vsplat_R(8);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, i8);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_q4_1_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1280);

        HVX_Vector v_sum = accum_4bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_sum_a   = v_act[9];

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);

        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q4_1_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1280);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1280);

        HVX_VectorPair v_sums = accum_4bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_sum_a_c0   = v_act0[9];
        HVX_Vector v_scale_a_c1 = v_act1[8];
        HVX_Vector v_sum_a_c1   = v_act1[9];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c0);
        HVX_Vector v_offset_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c1);
        HVX_Vector v_offset_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c1);

        HVX_Vector v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c0 = hvx_vec_add_f32_f32(v_scaled_dot_c0, v_offset_comb_c0);

        HVX_Vector v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_add_f32_f32(v_scaled_dot_c1, v_offset_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_q8_0_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 1152);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_q8_0_32x1(vptr, v_act);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[8];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q8_0_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 1152);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q8_0_32x2(vptr, v_act0, v_act1);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[8];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_q5_k_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 768);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1280);

        HVX_Vector v_sum = accum_5bit_32x1(vptr, v_act, Q6_V_vzero());
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_sum_a   = v_act[9];

        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a);
        HVX_Vector v_offset_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a);

        HVX_Vector v_scaled_dot = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);
        HVX_Vector v_sum_scaled = hvx_vec_add_f32_f32(v_scaled_dot, v_offset_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q5_k_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 768);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1280);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1280);

        HVX_VectorPair v_sums = accum_5bit_32x2(vptr, v_act0, v_act1, Q6_V_vzero());
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_offset = vptr[4];
        HVX_VectorPair p_deal = Q6_W_vdeal_VVR(v_scale_offset, v_scale_offset, -2);
        HVX_Vector v_scale = Q6_V_lo_W(p_deal);
        HVX_Vector v_offset = Q6_V_hi_W(p_deal);

        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_sum_a_c0   = v_act0[9];
        HVX_Vector v_scale_a_c1 = v_act1[8];
        HVX_Vector v_sum_a_c1   = v_act1[9];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c0);
        HVX_Vector v_offset_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale, v_scale_a_c1);
        HVX_Vector v_offset_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_offset, v_sum_a_c1);

        HVX_Vector v_scaled_dot_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c0 = hvx_vec_add_f32_f32(v_scaled_dot_c0, v_offset_comb_c0);

        HVX_Vector v_scaled_dot_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_add_f32_f32(v_scaled_dot_c1, v_offset_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_q6_k_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 896);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_VectorPair v_sums = accum_q6_k_32x1(vptr, v_act, i32);
        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, scale_q6_k_32x1(v_sums, vptr[6], v_act[8]));
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_q6_k_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector i32 = Q6_Vb_vsplat_R(32);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 896);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums0, v_sums1;
        accum_q6_k_32x2(vptr, v_act0, v_act1, i32, &v_sums0, &v_sums1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, scale_q6_k_32x1(v_sums0, vptr[6], v_act0[8]));
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, scale_q6_k_32x1(v_sums1, vptr[6], v_act1[8]));
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_iq4nl_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a = v_act[8];
        HVX_Vector v_scale_comb = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_iq4nl_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_iq4nl_lut;

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = vptr[4];
        HVX_Vector v_scale_a_c0 = v_act0[8];
        HVX_Vector v_scale_a_c1 = v_act1[8];

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f16_f16_to_f32_lower32(v_scale_w, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static void tiled_vec_dot_mxfp4_32x1(const uint32_t n, float * restrict s, const void * restrict vx, const void * restrict vy, uint32_t valid_rows, const float * restrict sz) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y_q = vy;

    HVX_Vector v_sum_float = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act = (const HVX_Vector *) (y_q + kt * 1152);

        HVX_Vector v_sum = accum_4bit_32x1_lut(vptr, v_act, mask_h4, lut);
        HVX_Vector v_sum_sf = Q6_Vsf_equals_Vw(v_sum);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_f16 = v_act[8];
        HVX_VectorPair p_scale_a_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_f16);
        HVX_Vector v_scale_a = Q6_V_lo_W(p_scale_a_f32);

        HVX_Vector v_scale_comb = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a);
        HVX_Vector v_sum_scaled = hvx_vec_mul_f32_f32(v_sum_sf, v_scale_comb);

        v_sum_float = hvx_vec_add_f32_f32(v_sum_float, v_sum_scaled);
    }

    v_sum_float = hvx_vec_mul_f32_f32(v_sum_float, hvx_vec_splat_f32(0.5f));

    if (sz) {
        hvx_vec_store_u(s, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float, hvx_vmemu(sz)));
    } else {
        hvx_vec_store_u(s, valid_rows * sizeof(float), v_sum_float);
    }
}

static void tiled_vec_dot_mxfp4_32x2(const uint32_t n, float * restrict s0, float * restrict s1, const void * restrict vx, const void * restrict vy0, const void * restrict vy1, uint32_t valid_rows, const float * restrict sz0, const float * restrict sz1) {
    const uint8_t * restrict tile_ptr = vx;
    const uint8_t * restrict y0_q = vy0;
    const uint8_t * restrict y1_q = vy1;

    HVX_Vector v_sum_float_c0 = Q6_V_vzero();
    HVX_Vector v_sum_float_c1 = Q6_V_vzero();
    HVX_Vector mask_h4 = Q6_Vb_vsplat_R(0x0F);
    HVX_Vector lut = *(const HVX_Vector *) kvalues_mxfp4_lut;
    HVX_Vector expand = *(const HVX_Vector *) expand_x32_e8m0;
    HVX_Vector e8m0_mask = Q6_V_vsplat_R(0x000000ff);

    uint32_t n_k_tiles = n / 32;
    for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
        const HVX_Vector * restrict vptr = (const HVX_Vector *) (tile_ptr + kt * 640);
        const HVX_Vector * restrict v_act0 = (const HVX_Vector *) (y0_q + kt * 1152);
        const HVX_Vector * restrict v_act1 = (const HVX_Vector *) (y1_q + kt * 1152);

        HVX_VectorPair v_sums = accum_4bit_32x2_lut(vptr, v_act0, v_act1, mask_h4, lut);
        HVX_Vector v_sum_c0 = Q6_V_lo_W(v_sums);
        HVX_Vector v_sum_c1 = Q6_V_hi_W(v_sums);

        HVX_Vector v_sum_sf_c0 = Q6_Vsf_equals_Vw(v_sum_c0);
        HVX_Vector v_sum_sf_c1 = Q6_Vsf_equals_Vw(v_sum_c1);

        HVX_Vector v_scale_w = hvx_vmem(tile_ptr + kt * 640 + 512);
        HVX_Vector r0_d = Q6_V_vdelta_VV(v_scale_w, expand);
        r0_d = Q6_V_vand_VV(r0_d, e8m0_mask);
        HVX_Vector v_scale_w_f32 = Q6_Vw_vasl_VwR(r0_d, 23);

        HVX_Vector v_scale_a_c0_f16 = v_act0[8];
        HVX_Vector v_scale_a_c1_f16 = v_act1[8];

        HVX_VectorPair p_scale_a_c0_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c0_f16);
        HVX_VectorPair p_scale_a_c1_f32 = hvx_vec_f16_to_f32_shuff(v_scale_a_c1_f16);

        HVX_Vector v_scale_a_c0 = Q6_V_lo_W(p_scale_a_c0_f32);
        HVX_Vector v_scale_a_c1 = Q6_V_lo_W(p_scale_a_c1_f32);

        HVX_Vector v_scale_comb_c0 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c0);
        HVX_Vector v_scale_comb_c1 = hvx_vec_mul_f32_f32(v_scale_w_f32, v_scale_a_c1);

        HVX_Vector v_sum_scaled_c0 = hvx_vec_mul_f32_f32(v_sum_sf_c0, v_scale_comb_c0);
        HVX_Vector v_sum_scaled_c1 = hvx_vec_mul_f32_f32(v_sum_sf_c1, v_scale_comb_c1);

        v_sum_float_c0 = hvx_vec_add_f32_f32(v_sum_float_c0, v_sum_scaled_c0);
        v_sum_float_c1 = hvx_vec_add_f32_f32(v_sum_float_c1, v_sum_scaled_c1);
    }

    v_sum_float_c0 = hvx_vec_mul_f32_f32(v_sum_float_c0, hvx_vec_splat_f32(0.5f));
    v_sum_float_c1 = hvx_vec_mul_f32_f32(v_sum_float_c1, hvx_vec_splat_f32(0.5f));

    if (sz0) {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c0, hvx_vmemu(sz0)));
    } else {
        hvx_vec_store_u(s0, valid_rows * sizeof(float), v_sum_float_c0);
    }
    if (sz1) {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), hvx_vec_add_f32_f32(v_sum_float_c1, hvx_vmemu(sz1)));
    } else {
        hvx_vec_store_u(s1, valid_rows * sizeof(float), v_sum_float_c1);
    }
}

static inline void quantize_f32_q8_0_tiled_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_size,
    size_t dst_row_size
) {
    (void) tmp_data;
    for (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_0_tiled((float *) src_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

static inline void quantize_f32_q8_1_tiled_kernel(
    const uint8_t * restrict src_data,
    uint8_t * restrict dst_data,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t nrows,
    size_t src_row_size,
    size_t dst_row_size
) {
    (void) tmp_data;
    for (uint32_t i = 0; i < nrows; ++i) {
        quantize_row_f32_q8_1_tiled((float *) src_data, dst_data, ne0);
        dst_data += dst_row_size;
        src_data += src_row_size;
    }
}

static inline void quantize_f32_q8_0_tiled_block_kernel(
    const float * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    size_t src_row_size,
    size_t dst_row_size,
    uint32_t r,
    uint32_t c
) {
    (void) tmp_data;
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne0 + qk - 1) / qk;

    for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        const float * restrict src_ptr = (const float *) ((const uint8_t *) src + r * src_row_size + c * qk * sizeof(float));
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1152;

        quantize_block_f32_q8_0_tiled((float *) src_ptr, dst_ptr);

        c++;
        if (c == nb) {
            c = 0;
            r++;
        }
    }
}

static inline void quantize_f32_q8_1_tiled_block_kernel(
    const float * restrict src,
    uint8_t * restrict dst,
    uint8_t * restrict tmp_data,
    uint32_t ne0,
    uint32_t ib_first,
    uint32_t ib_last,
    size_t src_row_size,
    size_t dst_row_size,
    uint32_t r,
    uint32_t c
) {
    (void) tmp_data;
    const uint32_t qk = QK_Q8_0_TILED;
    const uint32_t nb = (ne0 + qk - 1) / qk;

    for (uint32_t ib = ib_first; ib < ib_last; ++ib) {
        const float * restrict src_ptr = (const float *) ((const uint8_t *) src + r * src_row_size + c * qk * sizeof(float));
        uint8_t * restrict dst_ptr = dst + r * dst_row_size + c * 4 * 1280;

        quantize_block_f32_q8_1_tiled((float *) src_ptr, dst_ptr);

        c++;
        if (c == nb) {
            c = 0;
            r++;
        }
    }
}
