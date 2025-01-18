#ifndef EbDecCcso_h
#define EbDecCcso_h

#include "EbDecHandle.h"
#include "EbDecUtils.h"
#include "EbCcso.h"
// #include "Source/Lib/Encoder/Codec/EbPickccso.h"
// /home/Gabriel/CODEC/SVT-AV1_v2.1.0/Source/Lib/Encoder/Codec/EbPickccso.h
#ifdef __cplusplus
extern "C" {
#endif

typedef void (*dec_CCSO_FILTER_FUNC)(EbDecHandle * dec_handle, const int plane, const uint16_t *src_y,
                                 uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                 const uint8_t max_band_log2, const int edge_clf);

void dec_extend_ccso_border(uint16_t *buf, const int d, EbPictureBufferDesc* curbuf);
void dec_ccso_frame(EbPictureBufferDesc *frame, EbDecHandle * dec_handle, uint16_t *ext_rec_y);

// void svt_ccso_frame(EbDecHandle *dec_handle, int enable_flag);

// void svt_ccso_sb_row_mt(EbDecHandle *dec_handle, int32_t *mi_wide_l2, int32_t *mi_high_l2, uint16_t **colbuf,
//                         int32_t sb_fbr, uint16_t *src, int32_t *curr_recon_stride, uint8_t **curr_blk_recon_buf);

#ifdef __cplusplus
}
#endif
#endif // EbDecCcso_h_