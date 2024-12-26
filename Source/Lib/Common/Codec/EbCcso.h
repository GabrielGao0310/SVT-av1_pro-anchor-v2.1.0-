#ifndef EbCcso_h
#define EbCcso_h

#include <stdint.h>
#include "EbDefinitions.h"
#include "EbPictureControlSet.h"
#include "EbSequenceControlSet.h"
#include "EbAv1Structs.h"
#include "EbUtility.h"


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
                                 uint8_t *dst_yuv, const int dst_stride, const uint8_t thr, const uint8_t filter_sup,
                                 const uint8_t max_band_log2, const int edge_clf);

#ifdef __cplusplus
}
#endif
#endif // EbCcso_h