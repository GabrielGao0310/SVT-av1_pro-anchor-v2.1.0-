/*
* Copyright(c) 2019 Intel Corporation
* Copyright (c) 2016, Alliance for Open Media. All rights reserved
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/

#include <stdlib.h>
#include "aom_dsp_rtcd.h"
#include "EbDefinitions.h"
#include "EbEncHandle.h"
#include "EbCdefProcess.h"
#include "EbEncDecResults.h"
#include "EbThreads.h"
#include "EbReferenceObject.h"
#include "EbEncCdef.h"
#include "EbEncDecProcess.h"
#include "EbPictureBufferDesc.h"
#include "EbSequenceControlSet.h"
#include "EbUtility.h"
#include "EbPictureControlSet.h"
#include "EbResize.h"

#if CCSO
#include "EbPickccso.h"
#endif

void svt_av1_setup_dst_planes(PictureControlSet *pcs, struct MacroblockdPlane *planes, BlockSize bsize,
                              //const Yv12BufferConfig *src,
                              const EbPictureBufferDesc *src, int32_t mi_row, int32_t mi_col, const int32_t plane_start,
                              const int32_t plane_end);

void svt_aom_copy_sb8_16(uint16_t *dst, int32_t dstride, const uint8_t *src, int32_t src_voffset, int32_t src_hoffset,
                         int32_t sstride, int32_t vsize, int32_t hsize, Bool is_16bit);

void   *svt_aom_memalign(size_t align, size_t size);
void    svt_aom_free(void *memblk);
void   *svt_aom_malloc(size_t size);
int32_t svt_sb_all_skip(PictureControlSet *pcs, const Av1Common *const cm, int32_t mi_row, int32_t mi_col);
int32_t svt_sb_compute_cdef_list(PictureControlSet *pcs, const Av1Common *const cm, int32_t mi_row, int32_t mi_col,
                                 CdefList *dlist, BlockSize bs);
void    finish_cdef_search(PictureControlSet *pcs);
void    svt_av1_cdef_frame(SequenceControlSet *scs, PictureControlSet *pcs);
void    svt_av1_loop_restoration_save_boundary_lines(const Yv12BufferConfig *frame, Av1Common *cm, int32_t after_cdef);
void    svt_av1_superres_upscale_frame(struct Av1Common *cm, PictureControlSet *pcs, SequenceControlSet *scs);
void    set_unscaled_input_16bit(PictureControlSet *pcs);

void svt_aom_get_recon_pic(PictureControlSet *pcs, EbPictureBufferDesc **recon_ptr, Bool is_highbd);

/**************************************
 * Cdef Context
 **************************************/
typedef struct CdefContext {
    EbFifo *cdef_input_fifo_ptr;
    EbFifo *cdef_output_fifo_ptr;
} CdefContext;

static void cdef_context_dctor(EbPtr p) {
    EbThreadContext *thread_ctx = (EbThreadContext *)p;
    CdefContext     *obj        = (CdefContext *)thread_ctx->priv;
    EB_FREE_ARRAY(obj);
}

/******************************************************
 * Cdef Context Constructor
 ******************************************************/
EbErrorType svt_aom_cdef_context_ctor(EbThreadContext *thread_ctx, const EbEncHandle *enc_handle_ptr, int index) {
    CdefContext *cdef_ctx;
    EB_CALLOC_ARRAY(cdef_ctx, 1);
    thread_ctx->priv  = cdef_ctx;
    thread_ctx->dctor = cdef_context_dctor;

    // Input/Output System Resource Manager FIFOs
    cdef_ctx->cdef_input_fifo_ptr  = svt_system_resource_get_consumer_fifo(enc_handle_ptr->dlf_results_resource_ptr,
                                                                          index);
    cdef_ctx->cdef_output_fifo_ptr = svt_system_resource_get_producer_fifo(enc_handle_ptr->cdef_results_resource_ptr,
                                                                           index);

    return EB_ErrorNone;
}

#define default_mse_uv 1040400
static uint64_t compute_cdef_dist(const EbByte dst, int32_t doffset, int32_t dstride, const uint8_t *src,
                                  const CdefList *dlist, int32_t cdef_count, BlockSize bsize, int32_t coeff_shift,
                                  int32_t pli, uint8_t subsampling_factor, Bool is_16bit) {
    uint64_t curr_mse = 0;
    if (is_16bit) {
        curr_mse = svt_compute_cdef_dist_16bit(((uint16_t *)dst) + doffset,
                                               dstride,
                                               (uint16_t *)src,
                                               dlist,
                                               cdef_count,
                                               bsize,
                                               coeff_shift,
                                               pli,
                                               subsampling_factor);

    } else {
        curr_mse = svt_compute_cdef_dist_8bit(
            dst + doffset, dstride, src, dlist, cdef_count, bsize, coeff_shift, pli, subsampling_factor);
    }
    return curr_mse;
}

/* Search for the best filter strength pair for each 64x64 filter block.
 *
 * For each 64x64 filter block and each plane, search the allowable filter strength pairs.
 * Call cdef_filter_fb() to perform filtering, then compute the MSE for each pair.
*/
static void cdef_seg_search(PictureControlSet *pcs, SequenceControlSet *scs, uint32_t segment_index) {
    struct PictureParentControlSet *ppcs     = pcs->ppcs;
    FrameHeader                    *frm_hdr  = &ppcs->frm_hdr;
    Av1Common                      *cm       = ppcs->av1_cm;
    const Bool                      is_16bit = scs->is_16bit_pipeline;
    uint32_t                        x_seg_idx;
    uint32_t                        y_seg_idx;
    const uint32_t                  b64_pic_width  = (ppcs->aligned_width + 64 - 1) / 64;
    const uint32_t                  b64_pic_height = (ppcs->aligned_height + 64 - 1) / 64;
    SEGMENT_CONVERT_IDX_TO_XY(segment_index, x_seg_idx, y_seg_idx, pcs->cdef_segments_column_count);
    const uint32_t x_b64_start_idx = SEGMENT_START_IDX(x_seg_idx, b64_pic_width, pcs->cdef_segments_column_count);
    const uint32_t x_b64_end_idx   = SEGMENT_END_IDX(x_seg_idx, b64_pic_width, pcs->cdef_segments_column_count);
    const uint32_t y_b64_start_idx = SEGMENT_START_IDX(y_seg_idx, b64_pic_height, pcs->cdef_segments_row_count);
    const uint32_t y_b64_end_idx   = SEGMENT_END_IDX(y_seg_idx, b64_pic_height, pcs->cdef_segments_row_count);

    const int32_t mi_rows                    = cm->mi_rows;
    const int32_t mi_cols                    = cm->mi_cols;
    CdefControls *cdef_ctrls                 = &ppcs->cdef_ctrls;
    const int     first_pass_fs_num          = cdef_ctrls->first_pass_fs_num;
    const int     default_second_pass_fs_num = cdef_ctrls->default_second_pass_fs_num;
    EbByte        src[3];
    EbByte        ref[3];
    int32_t       stride_src[3];
    int32_t       stride_ref[3];
    int32_t       plane_bsize[3];
    int32_t       mi_wide_l2[3];
    int32_t       mi_high_l2[3];
    int32_t       xdec[3];
    int32_t       ydec[3];
    int32_t       cdef_count;
    const int32_t coeff_shift = AOMMAX(scs->static_config.encoder_bit_depth - 8, 0);
    const int32_t nvfb        = (mi_rows + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
    const int32_t nhfb        = (mi_cols + MI_SIZE_64X64 - 1) / MI_SIZE_64X64;
    const int32_t pri_damping = 3 + (frm_hdr->quantization_params.base_q_idx >> 6);
    const int32_t sec_damping = pri_damping;
    const int32_t num_planes  = 3;
    CdefList      dlist[MI_SIZE_128X128 * MI_SIZE_128X128];

    DECLARE_ALIGNED(32, uint16_t, inbuf[CDEF_INBUF_SIZE]);
    uint16_t *in = inbuf + CDEF_VBORDER * CDEF_BSTRIDE + CDEF_HBORDER;
    // tmp_dst is uint16_t to accomodate high bit depth content; 8bit will treat it as a uint8_t
    // buffer and will not use half of the buffer
    DECLARE_ALIGNED(32, uint16_t, tmp_dst[1 << (MAX_SB_SIZE_LOG2 * 2)]);

    EbPictureBufferDesc *input_pic = is_16bit ? pcs->input_frame16bit : ppcs->enhanced_pic;
    EbPictureBufferDesc *recon_pic;
    svt_aom_get_recon_pic(pcs, &recon_pic, is_16bit);

    for (int pli = 0; pli < num_planes; pli++) {
        const int subsampling_x = (pli == 0) ? 0 : 1;
        const int subsampling_y = (pli == 0) ? 0 : 1;
        xdec[pli]               = subsampling_x;
        ydec[pli]               = subsampling_y;
        plane_bsize[pli]        = subsampling_y ? (subsampling_x ? BLOCK_4X4 : BLOCK_8X4)
                                                : (subsampling_x ? BLOCK_4X8 : BLOCK_8X8);
        mi_wide_l2[pli]         = MI_SIZE_LOG2 - subsampling_x;
        mi_high_l2[pli]         = MI_SIZE_LOG2 - subsampling_y;
        src[pli]                = pcs->cdef_input_recon[pli];
        ref[pli]                = pcs->cdef_input_source[pli];
        stride_src[pli] = pli == 0 ? recon_pic->stride_y : (pli == 1 ? recon_pic->stride_cb : recon_pic->stride_cr);
        stride_ref[pli] = pli == 0 ? input_pic->stride_y : (pli == 1 ? input_pic->stride_cb : input_pic->stride_cr);
    }

    // Loop over all filter blocks (64x64)
    for (uint32_t fbr = y_b64_start_idx; fbr < y_b64_end_idx; ++fbr) {
        for (uint32_t fbc = x_b64_start_idx; fbc < x_b64_end_idx; ++fbc) {
            int32_t           dirinit = 0;
            const uint32_t    lc      = MI_SIZE_64X64 * fbc;
            const uint32_t    lr      = MI_SIZE_64X64 * fbr;
            int               nhb     = AOMMIN(MI_SIZE_64X64, mi_cols - lc);
            int               nvb     = AOMMIN(MI_SIZE_64X64, mi_rows - lr);
            int               hb_step = 1; //these should be all time with 64x64 SBs
            int               vb_step = 1;
            BlockSize         bs      = BLOCK_64X64;
            ModeInfo        **mi      = pcs->mi_grid_base + lr * cm->mi_stride + lc;
            const MbModeInfo *mbmi    = &mi[0]->mbmi;
            const BlockSize   bsize   = mbmi->block_mi.bsize;
            if (((fbc & 1) && (bsize == BLOCK_128X128 || bsize == BLOCK_128X64)) ||
                ((fbr & 1) && (bsize == BLOCK_128X128 || bsize == BLOCK_64X128)))
                continue;
            if (bsize == BLOCK_128X128 || bsize == BLOCK_128X64 || bsize == BLOCK_64X128)
                bs = bsize;

            if (bs == BLOCK_128X128 || bs == BLOCK_128X64) {
                nhb     = AOMMIN(MI_SIZE_128X128, cm->mi_cols - lc);
                hb_step = 2;
            }
            if (bs == BLOCK_128X128 || bs == BLOCK_64X128) {
                nvb     = AOMMIN(MI_SIZE_128X128, cm->mi_rows - lr);
                vb_step = 2;
            }
            const uint32_t fb_idx = fbr * nhfb + fbc;
            // No filtering if the entire filter block is skipped
            cdef_count = svt_sb_compute_cdef_list(pcs, cm, lr, lc, dlist, bs);
            if (cdef_count == 0) {
                pcs->skip_cdef_seg[fb_idx] = 1;
                continue;
            }
            pcs->skip_cdef_seg[fb_idx] = 0;

            uint8_t(*dir)[CDEF_NBLOCKS][CDEF_NBLOCKS] = &pcs->cdef_dir_data[fb_idx].dir;
            int32_t(*var)[CDEF_NBLOCKS][CDEF_NBLOCKS] = &pcs->cdef_dir_data[fb_idx].var;
            for (int pli = 0; pli < num_planes; pli++) {
                /* We avoid filtering the pixels for which some of the pixels to
                   average are outside the frame. We could change the filter instead,
                   but it would add special cases for any future vectorization.
                   No need to set pli == 2 because the copy size will be the same as for pli == 1. */
                if (pli < 2)
                    memset(inbuf, (uint8_t)CDEF_VERY_LARGE, sizeof(inbuf[0]) * CDEF_INBUF_SIZE);
                int32_t yoff  = CDEF_VBORDER * (fbr != 0);
                int32_t xoff  = CDEF_HBORDER * (fbc != 0);
                int32_t ysize = (nvb << mi_high_l2[pli]) + CDEF_VBORDER * ((int32_t)fbr + vb_step < nvfb) + yoff;
                int32_t xsize = (nhb << mi_wide_l2[pli]) + CDEF_HBORDER * ((int32_t)fbc + hb_step < nhfb) + xoff;

                svt_aom_copy_sb8_16(&in[(-yoff * CDEF_BSTRIDE - xoff)],
                                    CDEF_BSTRIDE,
                                    src[pli],
                                    (lr << mi_high_l2[pli]) - yoff,
                                    (lc << mi_wide_l2[pli]) - xoff,
                                    stride_src[pli],
                                    ysize,
                                    xsize,
                                    is_16bit);

                uint8_t subsampling_factor = cdef_ctrls->subsampling_factor;
                /*
                Cap the subsampling for certain block sizes.

                The intrinsics process several lines simultaneously, so blocks can only be subsampled
                a finite amount before there is no more speed gain.  If the space between processed lines
                is too large, the intrinsics will begin accessing memory outside the block.
                */
                switch (plane_bsize[pli]) {
                case BLOCK_8X8: subsampling_factor = MIN(subsampling_factor, 4); break;
                case BLOCK_8X4:
                case BLOCK_4X8: subsampling_factor = MIN(subsampling_factor, 2); break;
                case BLOCK_4X4: subsampling_factor = MIN(subsampling_factor, 1); break;
                }

                /* first cdef stage
                 * Perform the pri_filter strength search for the current sub_block
                 */
                for (int gi = 0; gi < first_pass_fs_num; gi++) {
                    // Check if chroma filter is set to be tested
                    if (pli && (cdef_ctrls->default_first_pass_fs_uv[gi] == -1)) {
                        pcs->mse_seg[1][fb_idx][gi] = default_mse_uv * 64;
                        continue;
                    }

                    int32_t pri_strength = cdef_ctrls->default_first_pass_fs[gi] / CDEF_SEC_STRENGTHS;
                    int32_t sec_strength = cdef_ctrls->default_first_pass_fs[gi] % CDEF_SEC_STRENGTHS;

                    svt_cdef_filter_fb(is_16bit ? NULL : (uint8_t *)tmp_dst,
                                       is_16bit ? tmp_dst : NULL,
                                       0,
                                       in,
                                       xdec[pli],
                                       ydec[pli],
                                       *dir,
                                       &dirinit,
                                       *var,
                                       pli,
                                       dlist,
                                       cdef_count,
                                       pri_strength,
                                       sec_strength + (sec_strength == 3),
                                       pri_damping,
                                       sec_damping,
                                       coeff_shift,
                                       subsampling_factor);
                    uint64_t curr_mse = compute_cdef_dist(
                        ref[pli],
                        (lr << mi_high_l2[pli]) * stride_ref[pli] + (lc << mi_wide_l2[pli]),
                        stride_ref[pli],
                        (uint8_t *)tmp_dst,
                        dlist,
                        cdef_count,
                        (BlockSize)plane_bsize[pli],
                        coeff_shift,
                        pli,
                        subsampling_factor,
                        is_16bit);

                    if (pli < 2)
                        pcs->mse_seg[pli][fb_idx][gi] = curr_mse * subsampling_factor;
                    else
                        pcs->mse_seg[1][fb_idx][gi] += (curr_mse * subsampling_factor);
                }

                /* second cdef stage
                 * Perform the sec_filter strength search for the current sub_block
                 */
                for (int gi = first_pass_fs_num; gi < first_pass_fs_num + default_second_pass_fs_num; gi++) {
                    // Check if chroma filter is set to be tested
                    if (pli && (cdef_ctrls->default_second_pass_fs_uv[gi - first_pass_fs_num] == -1)) {
                        pcs->mse_seg[1][fb_idx][gi] = default_mse_uv * 64;
                        continue;
                    }

                    int32_t pri_strength = cdef_ctrls->default_second_pass_fs[gi - first_pass_fs_num] /
                        CDEF_SEC_STRENGTHS;
                    int32_t sec_strength = cdef_ctrls->default_second_pass_fs[gi - first_pass_fs_num] %
                        CDEF_SEC_STRENGTHS;

                    svt_cdef_filter_fb(is_16bit ? NULL : (uint8_t *)tmp_dst,
                                       is_16bit ? tmp_dst : NULL,
                                       0,
                                       in,
                                       xdec[pli],
                                       ydec[pli],
                                       *dir,
                                       &dirinit,
                                       *var,
                                       pli,
                                       dlist,
                                       cdef_count,
                                       pri_strength,
                                       sec_strength + (sec_strength == 3),
                                       pri_damping,
                                       sec_damping,
                                       coeff_shift,
                                       subsampling_factor);
                    uint64_t curr_mse = compute_cdef_dist(
                        ref[pli],
                        (lr << mi_high_l2[pli]) * stride_ref[pli] + (lc << mi_wide_l2[pli]),
                        stride_ref[pli],
                        (uint8_t *)tmp_dst,
                        dlist,
                        cdef_count,
                        (BlockSize)plane_bsize[pli],
                        coeff_shift,
                        pli,
                        subsampling_factor,
                        is_16bit);

                    if (pli < 2)
                        pcs->mse_seg[pli][fb_idx][gi] = curr_mse * subsampling_factor;
                    else
                        pcs->mse_seg[1][fb_idx][gi] += (curr_mse * subsampling_factor);
                }
            }
        }
    }
}



/******************************************************
 * CDEF Kernel
 ******************************************************/
void *svt_aom_cdef_kernel(void *input_ptr) {
    // Context & SCS & PCS
    EbThreadContext    *thread_ctx  = (EbThreadContext *)input_ptr;
    CdefContext        *context_ptr = (CdefContext *)thread_ctx->priv;
    PictureControlSet  *pcs;

    //// Input
    EbObjectWrapper *dlf_results_wrapper;
    DlfResults      *dlf_results;

    //// Output
    EbObjectWrapper *cdef_results_wrapper = NULL;
    CdefResults     *cdef_results = NULL;

    // SB Loop variables

    for (;;) {

        // Get DLF Results
        EB_GET_FULL_OBJECT(context_ptr->cdef_input_fifo_ptr, &dlf_results_wrapper);

        dlf_results                   = (DlfResults *)dlf_results_wrapper->object_ptr;
        pcs                           = (PictureControlSet *)dlf_results->pcs_wrapper->object_ptr;

        FrameHeader *frm_hdr;
        SequenceControlSet *scs;

        PictureParentControlSet *ppcs = pcs->ppcs;
        scs                           = pcs->scs;

        Bool       is_16bit      = scs->is_16bit_pipeline;
        Av1Common *cm            = pcs->ppcs->av1_cm;
        frm_hdr                  = &pcs->ppcs->frm_hdr;
        CdefControls *cdef_ctrls = &pcs->ppcs->cdef_ctrls;
        
        // printf("\nccso\n");
    #if CCSO
        uint8_t* ref_yuv[3];
        EbPictureBufferDesc *ref      = is_16bit ? pcs->input_frame16bit : pcs->ppcs->enhanced_pic;
        const uint32_t       input_offset_y = ref->org_x + ref->org_y * ref->stride_y;
        ref_yuv[0]           = ref->buffer_y + (input_offset_y << is_16bit);
        const uint32_t input_offset_cb      = (ref->org_x + ref->org_y * ref->stride_cb) >> 1;
        ref_yuv[1]           = ref->buffer_cb + (input_offset_cb << is_16bit);
        const uint32_t input_offset_cr      = (ref->org_x + ref->org_y * ref->stride_cr) >> 1;
        ref_yuv[2]           = ref->buffer_cr + (input_offset_cr << is_16bit);

        uint16_t *rec_uv[CCSO_NUM_COMPONENTS];
        uint16_t *org_uv[CCSO_NUM_COMPONENTS];
        uint16_t *ext_rec_y = NULL;
        const int num_planes = av1_num_planes(&scs->seq_header.color_config);
        // printf("\nccso begin\n");
        EbPictureBufferDesc *recon_pic;
        svt_aom_get_recon_pic(pcs, &recon_pic, is_16bit);
        struct MacroblockdPlane pd[3];
        pd[0].subsampling_x = 0;
        pd[0].subsampling_y = 0;
        pd[0].plane_type    = PLANE_TYPE_Y;
        pd[0].is_16bit      = recon_pic->bit_depth > 8;
        pd[1].subsampling_x = 1;
        pd[1].subsampling_y = 1;
        pd[1].plane_type    = PLANE_TYPE_UV;
        pd[1].is_16bit      = recon_pic->bit_depth > 8;
        pd[2].subsampling_x = 1;
        pd[2].subsampling_y = 1;
        pd[2].plane_type    = PLANE_TYPE_UV;
        pd[2].is_16bit      = recon_pic->bit_depth > 8;
        if (pcs->ppcs->scs->is_16bit_pipeline)
            pd[0].is_16bit = pd[1].is_16bit = pd[2].is_16bit = TRUE;
        // /* pointer to current frame */
        // Yv12BufferConfig cur_buf;
        // svt_aom_link_eb_to_aom_buffer_desc_8bit(pcs->ppcs->enhanced_pic, &cur_buf);
        svt_av1_setup_dst_planes( pcs, pd, pcs->ppcs->scs->seq_header.sb_size, recon_pic, 0, 0, 0, num_planes);
        const int ccso_stride = pd[0].dst.width;
        const int ccso_stride_ext = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);

        for (int pli = 0; pli < num_planes; pli++) {
            rec_uv[pli] = svt_aom_malloc(sizeof(*rec_uv[pli]) * pd[0].dst.height *
                                    ccso_stride);
            org_uv[pli] = svt_aom_malloc(sizeof(*org_uv[pli]) * pd[0].dst.height *
                                    ccso_stride);
        }

        double mse1[3] = {0, 0, 0}; //rec和org
        double mse2[3] = {0, 0, 0}; //pd[pli].dst.buf和ref_yuv
        for (int p = 0; p < 3; p++){
            for(int i = 0; i < pd[p].dst.height; i++){
                for (int j = 0; j < pd[p].dst.width; j++) {
                    mse1[p] += (rec_uv[p][i * ccso_stride + j] - org_uv[p][i * ccso_stride + j]) * (rec_uv[p][i * ccso_stride + j] - org_uv[p][i * ccso_stride + j]);
                    int std = p==0 ? ref->stride_y : ref->stride_cb;
                    // uint8_t* buf = p==0 ? ref->buffer_y : (p==1 ? ref->buffer_cb : ref->buffer_cr);
                    uint8_t* buf = ref_yuv[p];
                    mse2[p] += (pd[p].dst.buf[i * pd[p].dst.stride + j] - buf[i * (std) + j]) * (pd[p].dst.buf[i * pd[p].dst.stride + j] - buf[i * (std) + j]);
                }
            }
            mse1[p] = mse1[p] / (pd[p].dst.height * pd[p].dst.width);
            mse2[p] = mse2[p] / (pd[p].dst.height * pd[p].dst.width);
        }

        ext_rec_y =
            svt_aom_malloc(sizeof(*ext_rec_y) *
                    (pd[0].dst.height + (CCSO_PADDING_SIZE << 1)) *
                    (pd[0].dst.width + (CCSO_PADDING_SIZE << 1)));
        for (int pli = 0; pli < 1; pli++) {
            const int pic_height = pd[pli].dst.height;
            const int pic_width = pd[pli].dst.width;
            const int dst_stride = pd[pli].dst.stride;
            for (int r = 0; r < pic_height; ++r) {
                for (int c = 0; c < pic_width; ++c) {
                    if (pli == 0) {
                        ext_rec_y[(r + CCSO_PADDING_SIZE) * ccso_stride_ext + c + CCSO_PADDING_SIZE] = (uint16_t)pd[pli].dst.buf[r * dst_stride + c];
                    }
                }
            }
        }
        extend_ccso_border(ext_rec_y, CCSO_PADDING_SIZE, pd);
    #endif

        if (!cdef_ctrls->use_reference_cdef_fs) {
            if (scs->seq_header.cdef_level && pcs->ppcs->cdef_level) {
                cdef_seg_search(pcs, scs, dlf_results->segment_index);
            }
        }
        //all seg based search is done. update total processed segments. if all done, finish the search and perfrom application.
        svt_block_on_mutex(pcs->cdef_search_mutex);

        pcs->tot_seg_searched_cdef++;
        if (pcs->tot_seg_searched_cdef == pcs->cdef_segments_total_count) {
            // SVT_LOG("    CDEF all seg here  %i\n", pcs->picture_number);
            if (scs->seq_header.cdef_level && pcs->ppcs->cdef_level) {
                finish_cdef_search(pcs);
                if (ppcs->enable_restoration || pcs->ppcs->is_ref || scs->static_config.recon_enabled) {
                    // Do application iff there are non-zero filters
                    if (frm_hdr->cdef_params.cdef_y_strength[0] != 0 || frm_hdr->cdef_params.cdef_uv_strength[0] != 0 ||
                        pcs->ppcs->nb_cdef_strengths != 1) {
                        svt_av1_cdef_frame(scs, pcs);
                    }
                }
            } else {
                frm_hdr->cdef_params.cdef_bits           = 0;
                frm_hdr->cdef_params.cdef_y_strength[0]  = 0;
                pcs->ppcs->nb_cdef_strengths             = 1;
                frm_hdr->cdef_params.cdef_uv_strength[0] = 0;
            }

    #if CCSO
            svt_aom_get_recon_pic(pcs, &recon_pic, is_16bit);
            svt_av1_setup_dst_planes( pcs, pd, pcs->ppcs->scs->seq_header.sb_size, recon_pic, 0, 0, 0, num_planes);

            uint16_t ref_stride;
            uint16_t *ref_buffer_16bit = NULL;
            uint8_t *ref_buffer_8bit = NULL;

            // Reading original and reconstructed chroma samples as input
            for (int pli = 0; pli < num_planes; pli++) {
                const int pic_height = pd[pli].dst.height;
                const int pic_width = pd[pli].dst.width;
                const int dst_stride = pd[pli].dst.stride;
                
                if (pcs->scs->is_16bit_pipeline) {
                    switch (pli) {
                        case 0:
                            ref_buffer_16bit = (uint16_t*)ref_yuv[pli];
                            ref_stride = ref->stride_y;
                            break;
                        case 1:
                            ref_buffer_16bit = (uint16_t*)ref_yuv[pli];
                            ref_stride = ref->stride_cb;
                            break;
                        case 2:
                            ref_buffer_16bit = (uint16_t*)ref_yuv[pli];
                            ref_stride = ref->stride_cr;
                            break;
                        default: ref_stride = 0;
                    }
                    for (int r = 0; r < pic_height; ++r) {
                        for (int c = 0; c < pic_width; ++c) {
                            rec_uv[pli][r * ccso_stride + c] = (uint16_t)pd[pli].dst.buf[r * dst_stride + c];
                            org_uv[pli][r * ccso_stride + c] = ref_buffer_16bit[r * ref_stride + c];
                        }
                    }
                } else {
                    // 8bit
                    switch (pli) {
                        case 0:{
                            // for (int i = 0; i < ref->luma_size; i++) {
                            //     ref_buffer[i] = (uint16_t)ref->buffer_y[i];
                            // }
                            ref_buffer_8bit = ref_yuv[pli];
                            ref_stride = ref->stride_y;
                            break;
                        }

                        case 1:{
                            // for (int i = 0; i < ref->chroma_size; i++) {
                            //     ref_buffer[i] = (uint16_t)ref->buffer_cb[i];
                            // }
                            ref_buffer_8bit = ref_yuv[pli];
                            ref_stride = ref->stride_cb;
                            break;
                        }

                        case 2:{
                            // for (int i = 0; i < ref->chroma_size; i++) {
                            //     ref_buffer[i] = (uint16_t)ref->buffer_cr[i];
                            // }
                            ref_buffer_8bit = ref_yuv[pli];
                            ref_stride = ref->stride_cr;
                            break;
                        }

                        default: ref_stride = 0;
                    }
                    for (int r = 0; r < pic_height; ++r) {
                        for (int c = 0; c < pic_width; ++c) {
                            rec_uv[pli][r * ccso_stride + c] = (uint16_t)pd[pli].dst.buf[r * dst_stride + c];
                            org_uv[pli][r * ccso_stride + c] = (uint16_t)ref_buffer_8bit[r * ref_stride + c];
                        }
                    }
                }
            }

            uint64_t   lambda;
            uint32_t   fast_lambda, full_lambda = 0;
            (*svt_aom_av1_lambda_assignment_function_table[pcs->ppcs->pred_structure])(
                pcs,
                &fast_lambda,
                &full_lambda,
                (uint8_t)pcs->ppcs->enhanced_pic->bit_depth,
                pcs->ppcs->frm_hdr.quantization_params.base_q_idx,
                FALSE);
            lambda   = full_lambda;
            // uint8_t   bit_depth   = pcs->hbd_md ? 10 : 8;

            // int32_t rdmult    = svt_aom_compute_rd_mult(pcs, ppcs->frm_hdr.quantization_params.base_q_idx, 
            // ppcs->frm_hdr.quantization_params.base_q_idx, bit_depth);

            // 算一下rec和org的sse，好像不大对劲
            double mse3[3] = {0, 0, 0}; //rec和org
            double mse4[3] = {0, 0, 0}; //pd[pli].dst.buf和ref_yuv
            for (int p = 0; p < 3; p++){
                for(int i = 0; i < pd[p].dst.height; i++){
                    for (int j = 0; j < pd[p].dst.width; j++) {
                        mse3[p] += (rec_uv[p][i * ccso_stride + j] - org_uv[p][i * ccso_stride + j]) * (rec_uv[p][i * ccso_stride + j] - org_uv[p][i * ccso_stride + j]);
                        int std = p==0 ? ref->stride_y : ref->stride_cb;
                        // uint8_t* buf = p==0 ? ref->buffer_y : (p==1 ? ref->buffer_cb : ref->buffer_cr);
                        uint8_t* buf = ref_yuv[p];
                        mse4[p] += (pd[p].dst.buf[i * pd[p].dst.stride + j] - buf[i * (std) + j]) * (pd[p].dst.buf[i * pd[p].dst.stride + j] - buf[i * (std) + j]);
                    }
                }
                mse3[p] = mse3[p] / (pd[p].dst.height * pd[p].dst.width);
                mse4[p] = mse4[p] / (pd[p].dst.height * pd[p].dst.width);
            }



            ccso_search(pcs, pd, (int)lambda, ext_rec_y, rec_uv, org_uv);
            ccso_frame(recon_pic, pcs, pd, ext_rec_y);


            // 算一下rec和org的sse，好像不大对劲
            double mse5[3] = {0, 0, 0}; //rec和org
            double mse6[3] = {0, 0, 0}; //pd[pli].dst.buf和ref_yuv
            for (int p = 0; p < 3; p++){
                for(int i = 0; i < pd[p].dst.height; i++){
                    for (int j = 0; j < pd[p].dst.width; j++) {
                        mse5[p] += (rec_uv[p][i * ccso_stride + j] - org_uv[p][i * ccso_stride + j]) * (rec_uv[p][i * ccso_stride + j] - org_uv[p][i * ccso_stride + j]);
                        int std = p==0 ? ref->stride_y : ref->stride_cb;
                        // uint8_t* buf = p==0 ? ref->buffer_y : (p==1 ? ref->buffer_cb : ref->buffer_cr);
                        uint8_t* buf = ref_yuv[p];
                        mse6[p] += (pd[p].dst.buf[i * pd[p].dst.stride + j] - buf[i * (std) + j]) * (pd[p].dst.buf[i * pd[p].dst.stride + j] - buf[i * (std) + j]);
                    }
                }
                mse5[p] = mse5[p] / (pd[p].dst.height * pd[p].dst.width);
                mse6[p] = mse6[p] / (pd[p].dst.height * pd[p].dst.width);

            }
            svt_aom_free(ext_rec_y);
            
            for (int pli = 0; pli < num_planes; pli++) {
                svt_aom_free(rec_uv[pli]);
                svt_aom_free(org_uv[pli]);
            }      
    #endif

            //restoration prep
            Bool is_lr = ppcs->enable_restoration && frm_hdr->allow_intrabc == 0;
            if (is_lr) {
                svt_av1_loop_restoration_save_boundary_lines(cm->frame_to_show, cm, 1);
                if (is_16bit) {
                    set_unscaled_input_16bit(pcs);
                }
            }

            // ------- start: Normative upscaling - super-resolution tool
            if (frm_hdr->allow_intrabc == 0 && pcs->ppcs->frame_superres_enabled) {
                svt_av1_superres_upscale_frame(cm, pcs, scs);
            }
            if (scs->static_config.resize_mode != RESIZE_NONE) {
                EbPictureBufferDesc *recon = NULL;
                svt_aom_get_recon_pic(pcs, &recon, is_16bit);
                recon->width  = pcs->ppcs->render_width;
                recon->height = pcs->ppcs->render_height;
                if (is_lr) {
                    EbPictureBufferDesc *input_pic = is_16bit ? pcs->input_frame16bit
                                                                : pcs->ppcs->enhanced_unscaled_pic;

                    svt_aom_assert_err(pcs->scaled_input_pic == NULL, "pcs_ptr->scaled_input_pic is not desctoried!");
                    EbPictureBufferDesc *scaled_input_pic = NULL;
                    // downscale input picture if recon is resized
                    Bool is_resized = recon->width != input_pic->width || recon->height != input_pic->height;
                    if (is_resized) {
                        superres_params_type spr_params = {recon->width, recon->height, 0};
                        svt_aom_downscaled_source_buffer_desc_ctor(&scaled_input_pic, input_pic, spr_params);
                        svt_aom_resize_frame(input_pic,
                                                scaled_input_pic,
                                                scs->static_config.encoder_bit_depth,
                                                av1_num_planes(&scs->seq_header.color_config),
                                                scs->subsampling_x,
                                                scs->subsampling_y,
                                                input_pic->packed_flag,
                                                PICTURE_BUFFER_DESC_FULL_MASK,
                                                0); // is_2bcompress
                        pcs->scaled_input_pic = scaled_input_pic;
                    }
                }
            }
            // ------- end: Normative upscaling - super-resolution tool

            pcs->rest_segments_column_count = scs->rest_segment_column_count;
            pcs->rest_segments_row_count    = scs->rest_segment_row_count;
            pcs->rest_segments_total_count = (uint16_t)(pcs->rest_segments_column_count * pcs->rest_segments_row_count);
            pcs->tot_seg_searched_rest     = 0;
            pcs->ppcs->av1_cm->use_boundaries_in_rest_search = scs->use_boundaries_in_rest_search;
            pcs->rest_extend_flag[0]                         = FALSE;
            pcs->rest_extend_flag[1]                         = FALSE;
            pcs->rest_extend_flag[2]                         = FALSE;

            uint32_t segment_index;
            for (segment_index = 0; segment_index < pcs->rest_segments_total_count; ++segment_index) {
                // Get Empty Cdef Results to Rest
                svt_get_empty_object(context_ptr->cdef_output_fifo_ptr, &cdef_results_wrapper);
                cdef_results                = (struct CdefResults *)cdef_results_wrapper->object_ptr;
                cdef_results->pcs_wrapper   = dlf_results->pcs_wrapper;
                cdef_results->segment_index = segment_index;
                // Post Cdef Results
                svt_post_full_object(cdef_results_wrapper);
            }
        }

        // cdef_and_ccso(pcs, dlf_results, context_ptr, cdef_results_wrapper, cdef_results);


        svt_release_mutex(pcs->cdef_search_mutex);

        // Release Dlf Results
        svt_release_object(dlf_results_wrapper);
    }

    return NULL;
}
