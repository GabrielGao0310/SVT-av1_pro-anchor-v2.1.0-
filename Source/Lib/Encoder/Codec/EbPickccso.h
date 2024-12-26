#ifndef EbPickccso_h
#define EbPickccso_h

#include "EbDefinitions.h"
#include "EbCcso.h"

#define CCSO_MAX_ITERATIONS 15

#define RDCOST_DBL_WITH_NATIVE_BD_DIST1(RM, R, D, BD) RDCOST_DBL((RM), (R), (double)((D) >> (2 * (BD - 8))))


#ifdef __cplusplus
extern "C" {
#endif



void ccso_search(PictureControlSet *pcs, MacroblockdPlane *pd, int rdmult, const uint16_t *ext_rec_y,
                 uint16_t *rec_uv[3], uint16_t *org_uv[3]);

/* Compute the aggregated residual between original and reconstructed sample for
* each entry of the LUT */
void ccso_pre_compute_class_err(MacroblockdPlane *pd, const int plane, const uint16_t *src_y, const uint16_t *ref,
                               const uint16_t *dst, uint8_t *src_cls0, uint8_t *src_cls1, const uint8_t shift_bits);

void ccso_pre_compute_class_err_bo(MacroblockdPlane *pd, const int plane, const uint16_t *src_y, const uint16_t *ref,
                                  const uint16_t *dst, const uint8_t shift_bits);

/* Apply CCSO on luma component at encoder (high bit-depth) */
void ccso_try_luma_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                         uint16_t *dst_yuv, const int dst_stride, const int8_t *filter_offset, uint8_t *src_cls0,
                         uint8_t *src_cls1, const uint8_t shift_bits, const uint8_t ccso_bo_only);

/* Apply CCSO on chroma component at encoder (high bit-depth) */
void ccso_try_chroma_filter(PictureControlSet *pcs, MacroblockdPlane *pd, const int plane, const uint16_t *src_y,
                           uint16_t *dst_yuv, const int dst_stride, const int8_t *filter_offset, uint8_t *src_cls0,
                           uint8_t *src_cls1, const uint8_t shift_bits, const uint8_t ccso_bo_only);

/* Derive the look-up table for a color component */
void derive_ccso_filter(PictureControlSet *pcs, const int plane, MacroblockdPlane *pd, const uint16_t *org_uv,
                       const uint16_t *ext_rec_y, const uint16_t *rec_uv, int rdmult);


/* Derive block level on/off for CCSO */
void derive_blk_md(Av1Common *cm, MacroblockdPlane *pd, const int plane, const uint64_t *unfiltered_dist,
                  const uint64_t *training_dist, bool *m_filter_control, uint64_t *cur_total_dist, int *cur_total_rate,
                  bool *filter_enable);

/* Compute the residual for each entry of the LUT using CCSO enabled filter
* blocks
*/
void ccso_compute_class_err(Av1Common *cm, const int plane, MacroblockdPlane *pd, const int max_band_log2,
                           const int max_edge_interval, const uint8_t ccso_bo_only);

/* Derive the offset value in the look-up table */
void derive_lut_offset(int8_t *temp_filter_offset, const int max_band_log2, const int max_edge_interval,
                      const uint8_t ccso_bo_only);

#ifdef __cplusplus
}
#endif
#endif // AV1_COMMON_CCSO_H_