#include "EbCcso.h"
#include "EbBitstreamUnit.h"
#include "common_dsp_rtcd.h"


void ccso_filter_block_hbd_with_buf_c(const uint16_t *src_y, uint16_t *dst_yuv, const uint8_t *src_cls0,
                                   const uint8_t *src_cls1, const int src_y_stride, const int dst_stride,
                                   const int src_cls_stride, const int x, const int y, const int pic_width,
                                   const int pic_height, const int8_t *filter_offset, const int blk_size,
                                   const int y_uv_hscale, const int y_uv_vscale, const int max_val,
                                   const uint8_t shift_bits, const uint8_t ccso_bo_only) {
   if (ccso_bo_only) {
       (void)src_cls0;
       (void)src_cls1;
   }
   int       cur_src_cls0;
   int       cur_src_cls1;
   const int y_end = AOMMIN(pic_height - y, blk_size);
   const int x_end = AOMMIN(pic_width - x, blk_size);
   for (int y_start = 0; y_start < y_end; y_start++) {
       const int y_pos = y_start;
       for (int x_start = 0; x_start < x_end; x_start++) {
           const int x_pos = x + x_start;
           if (!ccso_bo_only) {
               cur_src_cls0 = src_cls0[(y_pos << y_uv_vscale) * src_cls_stride + (x_pos << y_uv_hscale)];
               cur_src_cls1 = src_cls1[(y_pos << y_uv_vscale) * src_cls_stride + (x_pos << y_uv_hscale)];
           } else {
               cur_src_cls0 = 0;
               cur_src_cls1 = 0;
           }
           const int band_num    = src_y[(y_pos << y_uv_vscale) * src_y_stride + (x_pos << y_uv_hscale)] >> shift_bits;
           const int lut_idx_ext = (band_num << 4) + (cur_src_cls0 << 2) + cur_src_cls1;
           const int offset_val  = filter_offset[lut_idx_ext];
           dst_yuv[y_pos * dst_stride + x_pos] = clamp(offset_val + dst_yuv[y_pos * dst_stride + x_pos], 0, max_val);
       }
   }
}


void ccso_derive_src_block_c(const uint16_t *src_y, uint8_t *const src_cls0, uint8_t *const src_cls1,
                          const int src_y_stride, const int src_cls_stride, const int x, const int y,
                          const int pic_width, const int pic_height, const int y_uv_hscale, const int y_uv_vscale,
                          const int qstep, const int neg_qstep, const int *src_loc, const int blk_size,
                          const int edge_clf) {
   int       src_cls[2];
   const int y_end = AOMMIN(pic_height - y, blk_size);
   const int x_end = AOMMIN(pic_width - x, blk_size);
   for (int y_start = 0; y_start < y_end; y_start++) {
       const int y_pos = y_start;
       for (int x_start = 0; x_start < x_end; x_start++) {
           const int x_pos = x + x_start;
           cal_filter_support(src_cls,
                              &src_y[(y_pos << y_uv_vscale) * src_y_stride + (x_pos << y_uv_hscale)],
                              qstep,
                              neg_qstep,
                              src_loc,
                              edge_clf);
           src_cls0[(y_pos << y_uv_vscale) * src_cls_stride + (x_pos << y_uv_hscale)] = src_cls[0];
           src_cls1[(y_pos << y_uv_vscale) * src_cls_stride + (x_pos << y_uv_hscale)] = src_cls[1];
       }
   }
}

uint64_t compute_distortion_block_c(const uint16_t *org, const int org_stride, const uint16_t *rec16,
                                 const int rec_stride, const int x, const int y, const int log2_filter_unit_size,
                                 const int height, const int width) {
   int      err;
   uint64_t ssd = 0;
   int      y_offset;
   int      x_offset;
   if (y + (1 << log2_filter_unit_size) >= height)
       y_offset = height - y;
   else
       y_offset = (1 << log2_filter_unit_size);

   if (x + (1 << log2_filter_unit_size) >= width)
       x_offset = width - x;
   else
       x_offset = (1 << log2_filter_unit_size);

   for (int y_off = 0; y_off < y_offset; y_off++) {
       for (int x_off = 0; x_off < x_offset; x_off++) {
           err = org[org_stride * y_off + x + x_off] - rec16[rec_stride * y_off + x + x_off];
           ssd += err * err;
       }
   }
   return ssd;
}




static INLINE int32_t scaled_buffer_offset1(int32_t x_offset, int32_t y_offset, int32_t stride/*,
    const struct scale_factors *sf*/) {
    const int32_t x =
        /*sf ? sf->scale_value_x(x_offset, sf) >> SCALE_EXTRA_BITS :*/ x_offset;
    const int32_t y =
        /*sf ? sf->scale_value_y(y_offset, sf) >> SCALE_EXTRA_BITS :*/ y_offset;
    return y * stride + x;
}

static INLINE void setup_pred_plane1(struct Buf2D *dst, BlockSize bsize, uint8_t *src, int32_t width, int32_t height,
                                    int32_t stride, int32_t mi_row, int32_t mi_col,
                                    /*const struct scale_factors *scale,*/
                                    int32_t subsampling_x, int32_t subsampling_y, int32_t is_16bit) {
    // Offset the buffer pointer
    if (subsampling_y && (mi_row & 0x01) && (mi_size_high[bsize] == 1))
        mi_row -= 1;
    if (subsampling_x && (mi_col & 0x01) && (mi_size_wide[bsize] == 1))
        mi_col -= 1;

    const int32_t x = (MI_SIZE * mi_col) >> subsampling_x;
    const int32_t y = (MI_SIZE * mi_row) >> subsampling_y;
    dst->buf        = src + (scaled_buffer_offset1(x, y, stride /*, scale*/) << is_16bit);
    dst->buf0       = src;
    dst->width      = width;
    dst->height     = height;
    dst->stride     = stride;
}

void svt_av1_setup_dst_planes1(PictureControlSet *pcs, struct MacroblockdPlane *planes, BlockSize bsize,
                              //const Yv12BufferConfig *src,
                              const EbPictureBufferDesc *src, int32_t mi_row, int32_t mi_col, const int32_t plane_start,
                              const int32_t plane_end) {
    // We use AOMMIN(num_planes, MAX_MB_PLANE) instead of num_planes to quiet
    // the static analysis warnings.
    //for (int32_t i = plane_start; i < AOMMIN(plane_end, MAX_MB_PLANE); ++i) {
    //    struct MacroblockdPlane *const pd = &planes[i];
    //    const int32_t is_uv = i > 0;
    //    setup_pred_plane(&pd->dst, bsize, src->buffers[i], src->crop_widths[is_uv],
    //        src->crop_heights[is_uv], src->strides[is_uv], mi_row,
    //        mi_col, NULL, pd->subsampling_x, pd->subsampling_y);
    //}
    SequenceControlSet *scs = pcs->scs;
    for (int32_t i = plane_start; i < AOMMIN(plane_end, 3); ++i) {
        if (i == 0) {
            struct MacroblockdPlane *const pd = &planes[0];
            setup_pred_plane1(
                &pd->dst,
                bsize,
                &src->buffer_y[(src->org_x + src->org_y * src->stride_y) << pd->is_16bit],
                (scs->max_input_luma_width -
                 scs->max_input_pad_right), // The width/height should be the unpadded width/height (see AV1 spec 7.14.2 Edge Loop Filter Process)
                (scs->max_input_luma_height - scs->max_input_pad_bottom),
                src->stride_y,
                mi_row,
                mi_col,
                /*NULL,*/ pd->subsampling_x,
                pd->subsampling_y,
                pd->is_16bit); //AMIR: Updated to point to the right location
        } else if (i == 1) {
            struct MacroblockdPlane *const pd = &planes[1];
            setup_pred_plane1(
                &pd->dst,
                bsize,
                &src->buffer_cb[((src->org_x + src->org_y * src->stride_cb) << pd->is_16bit) / 2],
                (scs->max_input_luma_width - scs->max_input_pad_right) >>
                    1, // The width/height should be the unpadded width/height (see AV1 spec 7.14.2 Edge Loop Filter Process)
                (scs->max_input_luma_height - scs->max_input_pad_bottom) >> 1,
                src->stride_cb,
                mi_row,
                mi_col,
                /*NULL,*/ pd->subsampling_x,
                pd->subsampling_y,
                pd->is_16bit);
        } else if (i == 2) {
            struct MacroblockdPlane *const pd = &planes[2];
            setup_pred_plane1(
                &pd->dst,
                bsize,
                &src->buffer_cr[((src->org_x + src->org_y * src->stride_cr) << pd->is_16bit) / 2],
                (scs->max_input_luma_width - scs->max_input_pad_right) >>
                    1, // The width/height should be the unpadded width/height (see AV1 spec 7.14.2 Edge Loop Filter Process)
                (scs->max_input_luma_height - scs->max_input_pad_bottom) >> 1,
                src->stride_cr,
                mi_row,
                mi_col,
                /* NULL,*/ pd->subsampling_x,
                pd->subsampling_y,
                pd->is_16bit);
        }
    }
}
/* Pad the border of a frame */
void extend_ccso_border(uint16_t *buf, const int d, MacroblockdPlane *pd) {
    int       s = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
    uint16_t *p = &buf[d * s + d];
    int       h = pd[0].dst.height;
    int       w = pd[0].dst.width;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < d; x++) {
            *(p - d + x) = p[0];
            p[w + x]     = p[w - 1];
        }
        p += s;
    }
    p -= (s + d);
    for (int y = 0; y < d; y++) { memcpy(p + (y + 1) * s, p, sizeof(uint16_t) * (w + (d << 1))); }
    p -= ((h - 1) * s);
    for (int y = 0; y < d; y++) { memcpy(p - (y + 1) * s, p, sizeof(uint16_t) * (w + (d << 1))); }
}

/* Derive sample locations for CCSO */
void derive_ccso_sample_pos(int *rec_idx, const int ccsoStride, const uint8_t ext_filter_support) {
   // Input sample locations for CCSO
   //         2 1 4
   // 6 o 5 o 3 o 3 o 5 o 6
   //         4 1 2
   if (ext_filter_support == 0) {
       // Sample position 1
       rec_idx[0] = -1 * ccsoStride;
       rec_idx[1] = 1 * ccsoStride;
   } else if (ext_filter_support == 1) {
       // Sample position 2
       rec_idx[0] = -1 * ccsoStride - 1;
       rec_idx[1] = 1 * ccsoStride + 1;
   } else if (ext_filter_support == 2) {
       // Sample position 3
       rec_idx[0] = -1;
       rec_idx[1] = 1;
   } else if (ext_filter_support == 3) {
       // Sample position 4
       rec_idx[0] = 1 * ccsoStride - 1;
       rec_idx[1] = -1 * ccsoStride + 1;
   } else if (ext_filter_support == 4) {
       // Sample position 5
       rec_idx[0] = -3;
       rec_idx[1] = 3;
   } else { // if(ext_filter_support == 5) {
       // Sample position 6
       rec_idx[0] = -5;
       rec_idx[1] = 5;
   }
}

/* Derive the quantized index, later it can be used for retriving offset values
* from the look-up table */
void cal_filter_support(int *rec_luma_idx, const uint16_t *rec_y, const uint8_t quant_step_size,
                       const int inv_quant_step, const int *rec_idx, const int edge_clf) {
   if (edge_clf == 0) {
       for (int i = 0; i < 2; i++) {
           int d = rec_y[rec_idx[i]] - rec_y[0];
           if (d > quant_step_size)
               rec_luma_idx[i] = 2;
           else if (d < inv_quant_step)
               rec_luma_idx[i] = 0;
           else
               rec_luma_idx[i] = 1;
       }
   } else { // if (edge_clf == 1)
       for (int i = 0; i < 2; i++) {
           int d = rec_y[rec_idx[i]] - rec_y[0];
           if (d < inv_quant_step)
               rec_luma_idx[i] = 0;
           else
               rec_luma_idx[i] = 1;
       }
   }
}

void ccso_filter_block_hbd_wo_buf_c(const uint16_t *src_y, uint16_t *dst_yuv, const int x, const int y,
                                 const int pic_width, const int pic_height, int *src_cls, const int8_t *offset_buf,
                                 const int src_y_stride, const int dst_stride, const int y_uv_hscale,
                                 const int y_uv_vscale, const int thr, const int neg_thr, const int *src_loc,
                                 const int max_val, const int blk_size, const bool isSingleBand,
                                 const uint8_t shift_bits, const int edge_clf, const uint8_t ccso_bo_only) {
   const int y_end = AOMMIN(pic_height - y, blk_size);
   const int x_end = AOMMIN(pic_width - x, blk_size);
   for (int y_start = 0; y_start < y_end; y_start++) {
       const int y_pos = y_start;
       for (int x_start = 0; x_start < x_end; x_start++) {
           const int x_pos = x + x_start;
           if (!ccso_bo_only) {
               cal_filter_support(src_cls,
                                  &src_y[(y_pos << y_uv_vscale) * src_y_stride + (x_pos << y_uv_hscale)],
                                  thr,
                                  neg_thr,
                                  src_loc,
                                  edge_clf);
           } else {
               src_cls[0] = 0;
               src_cls[1] = 0;
           }
           const int band_num                  = isSingleBand
                                ? 0
                                : src_y[(y_pos << y_uv_vscale) * src_y_stride + (x_pos << y_uv_hscale)] >> shift_bits;
           const int lut_idx_ext               = (band_num << 4) + (src_cls[0] << 2) + src_cls[1];
           const int offset_val                = offset_buf[lut_idx_ext];
        //    Bool       is_16bit      = scs->is_16bit_pipeline;

           dst_yuv[y_pos * dst_stride + x_pos] = clamp(offset_val + dst_yuv[y_pos * dst_stride + x_pos], 0, max_val);
       }
   }
}

/* Apply CCSO on luma component when multiple bands are applied */
void ccso_apply_luma_mb_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                              uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                              const uint8_t max_band_log2, const int edge_clf) {
   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
   FrameHeader *frm_hdr = &pcs->ppcs->frm_hdr;

   const int     ccso_ext_stride = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
   const int     pic_height      = pd[plane].dst.height;
   const int     pic_width       = pd[plane].dst.width;
   const uint8_t shift_bits      = pcs->scs->static_config.encoder_bit_depth - max_band_log2;
   const int     max_val         = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_y)) * (y >> blk_log2) *
                   pcs->mi_stride +
               (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_x)) * (x >> blk_log2);
           const bool use_ccso = pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_y;
           if (!use_ccso)
               continue;
           if (frm_hdr->ccso_info.ccso_bo_only[plane]) {
               ccso_filter_block_hbd_wo_buf_c(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            0,
                                            0,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            false,
                                            shift_bits,
                                            edge_clf,
                                            frm_hdr->ccso_info.ccso_bo_only[plane]);
           } else {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            0,
                                            0,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            false,
                                            shift_bits,
                                            edge_clf,
                                            0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_ext_stride << blk_log2);
   }
}

/* Apply CCSO on luma component when single band is applied */
void ccso_apply_luma_sb_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                              uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                              const uint8_t max_band_log2, const int edge_clf) {
   (void)max_band_log2;
   FrameHeader *frm_hdr = &pcs->ppcs->frm_hdr;

   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
   const int ccso_ext_stride = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
   const int pic_height      = pd[plane].dst.height;
   const int pic_width       = pd[plane].dst.width;

   const uint8_t shift_bits = pcs->scs->static_config.encoder_bit_depth;
   const int     max_val    = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_y)) * (y >> blk_log2) *
                   pcs->mi_stride +
               (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_x)) * (x >> blk_log2);
           const bool use_ccso = pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_y;
           if (!use_ccso)
               continue;
           if (frm_hdr->ccso_info.ccso_bo_only[plane]) {
               ccso_filter_block_hbd_wo_buf_c(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            0,
                                            0,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            true,
                                            shift_bits,
                                            edge_clf,
                                            frm_hdr->ccso_info.ccso_bo_only[plane]);
           } else {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            0,
                                            0,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            true,
                                            shift_bits,
                                            edge_clf,
                                            0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_ext_stride << blk_log2);
   }
}

/* Apply CCSO on chroma component when multiple bands are applied */
void ccso_apply_chroma_mb_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                                uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                const uint8_t max_band_log2, const int edge_clf) {
   FrameHeader *frm_hdr = &pcs->ppcs->frm_hdr;

   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
   const int     ccso_ext_stride = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
   const int     pic_height      = pd[plane].dst.height;
   const int     pic_width       = pd[plane].dst.width;
   const int     y_uv_hscale     = pd[plane].subsampling_x;
   const int     y_uv_vscale     = pd[plane].subsampling_y;
   const uint8_t shift_bits      = pcs->scs->static_config.encoder_bit_depth - max_band_log2;
   const int     max_val         = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_y)) * (y >> blk_log2) *
                   pcs->mi_stride +
               (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_x)) * (x >> blk_log2);
           // pcs->mi_grid_base[(1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[1].subsampling_y)) * y_sb * pcs->mi_stride +
           //                     (1 << CCSO_BLK_SIZE >> (MI_SIZE_LOG2 - pd[1].subsampling_x)) * x_sb]
           //                         ->mbmi.ccso_blk_y = final_filter_control[y_sb * ccso_nhfb + x_sb];
           const bool use_ccso = (plane == 1) ? pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_u
                                              : pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_v;
           if (!use_ccso)
               continue;
           if (frm_hdr->ccso_info.ccso_bo_only[plane]) {
               ccso_filter_block_hbd_wo_buf_c(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            y_uv_hscale,
                                            y_uv_vscale,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            false,
                                            shift_bits,
                                            edge_clf,
                                            frm_hdr->ccso_info.ccso_bo_only[plane]);
           } else {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            y_uv_hscale,
                                            y_uv_vscale,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            false,
                                            shift_bits,
                                            edge_clf,
                                            0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_ext_stride << (blk_log2 + y_uv_vscale));
   }
}




/* Apply CCSO on chroma component when single bands is applied */
void ccso_apply_chroma_sb_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                                uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                const uint8_t max_band_log2, const int edge_clf) {
   (void)max_band_log2;
   FrameHeader *frm_hdr = &pcs->ppcs->frm_hdr;

   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
   const int     ccso_ext_stride = pd[0].dst.width + (CCSO_PADDING_SIZE << 1);
   const int     pic_height      = pd[plane].dst.height;
   const int     pic_width       = pd[plane].dst.width;
   const uint8_t shift_bits      = pcs->scs->static_config.encoder_bit_depth;
   const int     y_uv_hscale     = pd[plane].subsampling_x;
   const int     y_uv_vscale     = pd[plane].subsampling_y;
   const int     max_val         = (1 << pcs->scs->static_config.encoder_bit_depth) - 1;
   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_y)) * (y >> blk_log2) *
                   pcs->mi_stride +
               (blk_size >> (MI_SIZE_LOG2 - pd[plane].subsampling_x)) * (x >> blk_log2);
           const bool use_ccso = (plane == 1) ? pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_u
                                              : pcs->mi_grid_base[ccso_blk_idx]->mbmi.ccso_blk_v;

           if (!use_ccso)
               continue;
           if (frm_hdr->ccso_info.ccso_bo_only[plane]) {
               ccso_filter_block_hbd_wo_buf_c(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            y_uv_hscale,
                                            y_uv_vscale,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            true,
                                            shift_bits,
                                            edge_clf,
                                            frm_hdr->ccso_info.ccso_bo_only[plane]);
           } else {
               ccso_filter_block_hbd_wo_buf(src_y,
                                            dst_yuv,
                                            x,
                                            y,
                                            pic_width,
                                            pic_height,
                                            src_cls,
                                            frm_hdr->ccso_info.filter_offset[plane],
                                            ccso_ext_stride,
                                            dst_stride,
                                            y_uv_hscale,
                                            y_uv_vscale,
                                            thr,
                                            neg_thr,
                                            src_loc,
                                            max_val,
                                            blk_size,
                                            true,
                                            shift_bits,
                                            edge_clf,
                                            0);
           }
       }
       dst_yuv += (dst_stride << blk_log2);
       src_y += (ccso_ext_stride << (blk_log2 + y_uv_vscale));
   }
}

/* Apply CCSO for one frame */
// EbPictureBufferDesc *recon_pic
void ccso_frame(EbPictureBufferDesc *frame, PictureControlSet *pcs, MacroblockdPlane *pd, uint16_t *ext_rec_y) {
//    struct PictureParentControlSet *ppcs    = pcs->ppcs;
   FrameHeader                    *frm_hdr = &pcs->ppcs->frm_hdr;
//    Av1Common                      *cm      = ppcs->av1_cm;

   const int num_planes = av1_num_planes(&pcs->scs->seq_header.color_config);
   svt_av1_setup_dst_planes1(pcs, pd, pcs->ppcs->scs->seq_header.sb_size, frame, 0, 0, 0, num_planes);



   const uint8_t quant_sz[4] = {16, 8, 32, 64};
   for (int plane = 0; plane < num_planes; plane++) {
        // 把pd[plane].dst.buf放进16位中，处理完了再放回去
        const int pic_height = pd[plane].dst.height;
        const int pic_width = pd[plane].dst.width;
        const int dst_stride = pd[plane].dst.stride;
        uint16_t* dst_yuv16bit = (uint16_t *)malloc(dst_stride * pic_height * sizeof(uint16_t));

        for (int r = 0; r < pic_height; ++r) {
            for (int c = 0; c < pic_width; ++c) {
                dst_yuv16bit[r * dst_stride + c] = (uint16_t)pd[plane].dst.buf[r * dst_stride + c];
            }
        }



        // const int     dst_stride      = pd[plane].dst.stride;
        const uint8_t quant_step_size = quant_sz[frm_hdr->ccso_info.quant_idx[plane]];
        if (frm_hdr->ccso_info.ccso_enable[plane]) {
            CCSO_FILTER_FUNC apply_ccso_filter_func = frm_hdr->ccso_info.max_band_log2[plane]
                ? (plane > 0 ? ccso_apply_chroma_mb_filter : ccso_apply_luma_mb_filter)
                : (plane > 0 ? ccso_apply_chroma_sb_filter : ccso_apply_luma_sb_filter);
            apply_ccso_filter_func(pcs,
                                    pd,
                                    plane,
                                    ext_rec_y,
                                    // &(pd[plane].dst.buf)[0],
                                    dst_yuv16bit,
                                    dst_stride,
                                    quant_step_size,
                                    frm_hdr->ccso_info.ext_filter_support[plane],
                                    frm_hdr->ccso_info.max_band_log2[plane],
                                    frm_hdr->ccso_info.edge_clf[plane]);
        }

        for (int r = 0; r < pic_height; ++r) {
            for (int c = 0; c < pic_width; ++c) {
               pd[plane].dst.buf[r * dst_stride + c] = (uint8_t)dst_yuv16bit[r * dst_stride + c];
            }
        }
        free(dst_yuv16bit);
        dst_yuv16bit = NULL;
   }
}