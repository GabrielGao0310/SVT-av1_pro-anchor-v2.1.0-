#ifndef EbCcso_h
#define EbCcso_h

#include <stdint.h>
#include "EbDefinitions.h"
#include "Source/Lib/Encoder/Codec/EbPictureControlSet.h"
// /home/Gabriel/CODEC/SVT-AV1_v2.1.0/Source/Lib/Encoder/Codec/EbPictureControlSet.h
#include "Source/Lib/Encoder/Codec/EbSequenceControlSet.h"
#include "EbAv1Structs.h"
#include "EbUtility.h"
#define CCSO 0


#define CCSO_INPUT_INTERVAL 3
// Only need this for fixed-size arrays, for structs just assign.
#define av1_copy(dest, src)                  \
    {                                        \
        assert(sizeof(dest) == sizeof(src)); \
        memcpy(dest, src, sizeof(src));      \
    }

static const int edge_clf_to_edge_interval[2] = { 3, 2 };


#ifdef __cplusplus
extern "C" {
#endif

void extend_ccso_border(uint16_t *buf, const int d, MacroblockdPlane *pd);

/* Derive sample locations for CCSO */
void derive_ccso_sample_pos(int *rec_idx, const int ccsoStride, const uint8_t ext_filter_support);

/* Derive the quantized index, later it can be used for retriving offset values
* from the look-up table */
void cal_filter_support(int *rec_luma_idx, const uint16_t *rec_y, const uint8_t quant_step_size,
                       const int inv_quant_step, const int *rec_idx, const int edge_clf);

void ccso_frame(EbPictureBufferDesc *frame, PictureControlSet *pcs, MacroblockdPlane *pd, uint16_t *ext_rec_y);


/* Apply CCSO on one color component */
typedef void (*CCSO_FILTER_FUNC)(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                                 uint16_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                 const uint8_t max_band_log2, const int edge_clf);

void ccso_filter_block_hbd_with_buf_c(const uint16_t *src_y, uint16_t *dst_yuv, const uint8_t *src_cls0,
                                   const uint8_t *src_cls1, const int src_y_stride, const int dst_stride,
                                   const int src_cls_stride, const int x, const int y, const int pic_width,
                                   const int pic_height, const int8_t *filter_offset, const int blk_size,
                                   const int y_uv_hscale, const int y_uv_vscale, const int max_val,
                                   const uint8_t shift_bits, const uint8_t ccso_bo_only);

void ccso_derive_src_block_c(const uint16_t *src_y, uint8_t *const src_cls0, uint8_t *const src_cls1,
                          const int src_y_stride, const int src_cls_stride, const int x, const int y,
                          const int pic_width, const int pic_height, const int y_uv_hscale, const int y_uv_vscale,
                          const int qstep, const int neg_qstep, const int *src_loc, const int blk_size,
                          const int edge_clf);

uint64_t compute_distortion_block_c(const uint16_t *org, const int org_stride, const uint16_t *rec16,
                                 const int rec_stride, const int x, const int y, const int log2_filter_unit_size,
                                 const int height, const int width);

#ifdef __cplusplus
}
#endif
#endif // EbCcso_h