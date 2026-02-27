#ifndef TIMING_H
#define TIMING_H

#include <stdint.h>

#ifdef BARE_METAL
#ifndef CPU_MHZ
#define CPU_MHZ 100
#endif
#endif

#define YOLO_TIMING_OP_MAX  16
#define YOLO_TIMING_ENTRIES 512

void yolo_timing_set_layer(int layer_id);
void yolo_timing_begin(const char* op);
void yolo_timing_end(void);
void yolo_timing_end_with_op(const char* op);
void yolo_timing_print_layer_ops(int layer_id);
void yolo_timing_reset(void);

#endif /* TIMING_H */
