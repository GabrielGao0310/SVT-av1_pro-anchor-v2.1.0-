

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "EbDecUtils.h"
#include "EbDecNbr.h"
#include "EbUtility.h"
#include "EbDecCcso.h"


/* Pad the border of a frame */
void dec_extend_ccso_border(uint16_t *buf, const int d, EbPictureBufferDesc* curbuf) {
    int       s = curbuf->width + (CCSO_PADDING_SIZE << 1);
    uint16_t *p = &buf[d * s + d];
    int       h = curbuf->height;
    int       w = curbuf->width;
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




/* Apply CCSO on luma component when multiple bands are applied */
void dec_ccso_apply_luma_mb_filter(EbDecHandle * dec_handle, const int plane, const uint16_t *src_y,
                              uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                              const uint8_t max_band_log2, const int edge_clf) {
   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
    FrameHeader  *frm_hdr = &dec_handle->frame_header;
    EbPictureBufferDesc *frame = dec_handle->cur_pic_buf[0]->ps_pic_buf;
    MainFrameBuf        *main_frame_buf = &dec_handle->main_frame_buf;
    CurFrameBuf         *frame_buf      = &main_frame_buf->cur_frame_bufs[0];

   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
    const int     ccso_ext_stride = frame->width + (CCSO_PADDING_SIZE << 1);
    const int     y_uv_hscale     = (plane == 0) ? 0 : dec_handle->seq_header.color_config.subsampling_x;
    const int     y_uv_vscale     = (plane == 0) ? 0 : dec_handle->seq_header.color_config.subsampling_y;
    const int     pic_height      = (plane == 0) ? frame->height : frame->height >> y_uv_vscale;
    const int     pic_width       = (plane == 0) ? frame->width : frame->width >> y_uv_hscale;
    const uint8_t shift_bits      = frame->bit_depth;
    const int     max_val         = (1 << frame->bit_depth) - 1;
   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - y_uv_vscale)) * (y >> blk_log2) * frm_hdr->mi_stride + (blk_size >> (MI_SIZE_LOG2 - y_uv_hscale)) * (x >> blk_log2);

            SBInfo *sb_info = NULL;
            sb_info = frame_buf->sb_info + ccso_blk_idx;
           const bool use_ccso = (plane == 1) ? *(sb_info->sb_ccso_blk_u) : *(sb_info->sb_ccso_blk_v);
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
void dec_ccso_apply_luma_sb_filter(EbDecHandle * dec_handle, const int plane, const uint16_t *src_y,
                              uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                              const uint8_t max_band_log2, const int edge_clf) {
   (void)max_band_log2;
    FrameHeader  *frm_hdr = &dec_handle->frame_header;
    EbPictureBufferDesc *frame = dec_handle->cur_pic_buf[0]->ps_pic_buf;
    MainFrameBuf        *main_frame_buf = &dec_handle->main_frame_buf;
    CurFrameBuf         *frame_buf      = &main_frame_buf->cur_frame_bufs[0];

   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
    const int     ccso_ext_stride = frame->width + (CCSO_PADDING_SIZE << 1);
    const int     y_uv_hscale     = (plane == 0) ? 0 : dec_handle->seq_header.color_config.subsampling_x;
    const int     y_uv_vscale     = (plane == 0) ? 0 : dec_handle->seq_header.color_config.subsampling_y;
    const int     pic_height      = (plane == 0) ? frame->height : frame->height >> y_uv_vscale;
    const int     pic_width       = (plane == 0) ? frame->width : frame->width >> y_uv_hscale;
    const uint8_t shift_bits      = frame->bit_depth;
    const int     max_val         = (1 << frame->bit_depth) - 1;

   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - y_uv_vscale)) * (y >> blk_log2) * frm_hdr->mi_stride + (blk_size >> (MI_SIZE_LOG2 - y_uv_hscale)) * (x >> blk_log2);

            SBInfo *sb_info = NULL;
            sb_info = frame_buf->sb_info + ccso_blk_idx;
           const bool use_ccso = (plane == 1) ? *(sb_info->sb_ccso_blk_u) : *(sb_info->sb_ccso_blk_v);

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
void dec_ccso_apply_chroma_mb_filter(EbDecHandle * dec_handle, const int plane, const uint16_t *src_y,
                                uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                const uint8_t max_band_log2, const int edge_clf) {
    FrameHeader  *frm_hdr = &dec_handle->frame_header;
    EbPictureBufferDesc *frame = dec_handle->cur_pic_buf[0]->ps_pic_buf;
    MainFrameBuf        *main_frame_buf = &dec_handle->main_frame_buf;
    CurFrameBuf         *frame_buf      = &main_frame_buf->cur_frame_bufs[0];

   //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
    const int     ccso_ext_stride = frame->width + (CCSO_PADDING_SIZE << 1);
    const int     y_uv_hscale     = (plane == 0) ? 0 : dec_handle->seq_header.color_config.subsampling_x;
    const int     y_uv_vscale     = (plane == 0) ? 0 : dec_handle->seq_header.color_config.subsampling_y;
    const int     pic_height      = (plane == 0) ? frame->height : frame->height >> y_uv_vscale;
    const int     pic_width       = (plane == 0) ? frame->width : frame->width >> y_uv_hscale;
    const uint8_t shift_bits      = frame->bit_depth;
    const int     max_val         = (1 << frame->bit_depth) - 1;

   int           src_cls[2];
   const int     neg_thr = thr * -1;
   int           src_loc[2];
   derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
   const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
   const int blk_size = 1 << blk_log2;
   src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - y_uv_vscale)) * (y >> blk_log2) * frm_hdr->mi_stride + (blk_size >> (MI_SIZE_LOG2 - y_uv_hscale)) * (x >> blk_log2);

            SBInfo *sb_info = NULL;
            sb_info = frame_buf->sb_info + ccso_blk_idx;
           const bool use_ccso = (plane == 1) ? *(sb_info->sb_ccso_blk_u) : *(sb_info->sb_ccso_blk_v);
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
void dec_ccso_apply_chroma_sb_filter(EbDecHandle * dec_handle, const int plane, const uint16_t *src_y,
                                uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                const uint8_t max_band_log2, const int edge_clf) {
    (void)max_band_log2;
    FrameHeader  *frm_hdr = &dec_handle->frame_header;
    EbPictureBufferDesc *frame = dec_handle->cur_pic_buf[0]->ps_pic_buf;
    MainFrameBuf        *main_frame_buf = &dec_handle->main_frame_buf;
    CurFrameBuf         *frame_buf      = &main_frame_buf->cur_frame_bufs[0];
    //   const CommonModeInfoParams *const mi_params = &cm->mi_params;
    const int     ccso_ext_stride = frame->width + (CCSO_PADDING_SIZE << 1);
    const int     y_uv_hscale     = (plane == 0) ? 0 : dec_handle->seq_header.color_config.subsampling_x;
    const int     y_uv_vscale     = (plane == 0) ? 0 : dec_handle->seq_header.color_config.subsampling_y;
    const int     pic_height      = (plane == 0) ? frame->height : frame->height >> y_uv_vscale;
    const int     pic_width       = (plane == 0) ? frame->width : frame->width >> y_uv_hscale;
    const uint8_t shift_bits      = frame->bit_depth;

    const int     max_val         = (1 << frame->bit_depth) - 1;
    int           src_cls[2];
    const int     neg_thr = thr * -1;
    int           src_loc[2];
    derive_ccso_sample_pos(src_loc, ccso_ext_stride, filter_sup);
    const int blk_log2 = plane > 0 ? CCSO_BLK_SIZE : CCSO_BLK_SIZE + 1;
    const int blk_size = 1 << blk_log2;
    src_y += CCSO_PADDING_SIZE * ccso_ext_stride + CCSO_PADDING_SIZE;
   for (int y = 0; y < pic_height; y += blk_size) {
       for (int x = 0; x < pic_width; x += blk_size) {
           const int ccso_blk_idx = (blk_size >> (MI_SIZE_LOG2 - y_uv_vscale)) * (y >> blk_log2) * frm_hdr->mi_stride + (blk_size >> (MI_SIZE_LOG2 - y_uv_hscale)) * (x >> blk_log2);
            SBInfo *sb_info = NULL;
            sb_info = frame_buf->sb_info + ccso_blk_idx;
           const bool use_ccso = (plane == 1) ? *(sb_info->sb_ccso_blk_u) : *(sb_info->sb_ccso_blk_v);


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
void dec_ccso_frame(EbPictureBufferDesc *frame, EbDecHandle * dec_handle, uint16_t *ext_rec_y) {
    FrameHeader  *frm_hdr = &dec_handle->frame_header;
    const int32_t num_planes = av1_num_planes(&dec_handle->seq_header.color_config);

    // svt_av1_setup_dst_planes1(pcs, pd, dec_handle->seq_header.sb_size, frame, 0, 0, 0, num_planes);

   const uint8_t quant_sz[4] = {16, 8, 32, 64};
   for (int plane = 0; plane < num_planes; plane++) {
        // 把pd[plane].dst.buf放进16位中，处理完了再放回去
        const int pic_height = (plane == 0) ? frame->height : (frame->height / 2);
        const int pic_width = (plane == 0) ? frame->width : (frame->width / 2);
        const int dst_stride = (plane == 0) ? frame->stride_y : ((plane == 1) ? frame->stride_cb : frame->stride_cr);
        uint16_t* dst_yuv16bit = (uint16_t *)malloc(dst_stride * pic_height * sizeof(uint16_t));
        uint8_t* frame_buf = (plane == 0) ? frame->buffer_y : ((plane == 1) ? frame->buffer_cb : frame->buffer_cr);
        for (int r = 0; r < pic_height; ++r) {
            for (int c = 0; c < pic_width; ++c) {
                dst_yuv16bit[r * dst_stride + c] = (uint16_t)frame_buf[r * dst_stride + c];
            }
        }

        const uint8_t quant_step_size = quant_sz[frm_hdr->ccso_info.quant_idx[plane]];
        if (frm_hdr->ccso_info.ccso_enable[plane]) {
            dec_CCSO_FILTER_FUNC apply_ccso_filter_func = frm_hdr->ccso_info.max_band_log2[plane]
                ? (plane > 0 ? dec_ccso_apply_chroma_mb_filter : dec_ccso_apply_luma_mb_filter)
                : (plane > 0 ? dec_ccso_apply_chroma_sb_filter : dec_ccso_apply_luma_sb_filter);
            apply_ccso_filter_func(dec_handle,
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
               frame_buf[r * dst_stride + c] = (uint8_t)dst_yuv16bit[r * dst_stride + c];
            }
        }
        free(dst_yuv16bit);
        dst_yuv16bit = NULL;
   }
}