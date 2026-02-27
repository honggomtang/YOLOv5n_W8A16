#include "timing.h"
#include "mcycle.h"
#include <string.h>

#ifdef BARE_METAL
#include "../platform_config.h"
#include "xil_printf.h"
#ifndef CPU_MHZ
#define CPU_MHZ 100
#endif
#define TIMING_LOG(...) xil_printf(__VA_ARGS__)
#else
#include <stdio.h>
#define TIMING_LOG(...) printf(__VA_ARGS__)
#endif

typedef struct {
    int      layer;
    char     op[YOLO_TIMING_OP_MAX];
    uint64_t cycles;
} timing_entry_t;

static timing_entry_t s_entries[YOLO_TIMING_ENTRIES];
static int            s_count;
static int            s_cursor;
static int            s_current_layer;
static uint64_t       s_start;
static char           s_current_op[YOLO_TIMING_OP_MAX];

void yolo_timing_set_layer(int layer_id) {
    s_current_layer = layer_id;
}

void yolo_timing_begin(const char* op) {
    size_t len = 0;
    if (op) {
        while (op[len] && len < (size_t)(YOLO_TIMING_OP_MAX - 1))
            s_current_op[len] = op[len], len++;
    }
    s_current_op[len] = '\0';
    s_start = timer_read64();
}

void yolo_timing_end(void) {
    yolo_timing_end_with_op(NULL);
}

void yolo_timing_end_with_op(const char* op) {
    if (s_count >= YOLO_TIMING_ENTRIES) return;
    uint64_t delta = timer_delta64(s_start, timer_read64());
    s_entries[s_count].layer = s_current_layer;
    if (op && op[0]) {
        size_t len = 0;
        while (op[len] && len < (size_t)(YOLO_TIMING_OP_MAX - 1))
            s_entries[s_count].op[len] = op[len], len++;
        s_entries[s_count].op[len] = '\0';
    } else {
        (void)strncpy(s_entries[s_count].op, s_current_op, YOLO_TIMING_OP_MAX - 1);
        s_entries[s_count].op[YOLO_TIMING_OP_MAX - 1] = '\0';
    }
    s_entries[s_count].cycles = delta;
    s_count++;
}

void yolo_timing_print_layer_ops(int layer_id) {
    int i = s_cursor;
    while (i < s_count && s_entries[i].layer != layer_id) i++;
    if (i >= s_count) { s_cursor = s_count; return; }

    TIMING_LOG("    ");
    int first = 1;
    for (; i < s_count && s_entries[i].layer == layer_id; i++) {
        const char* op = s_entries[i].op;
        uint64_t c = s_entries[i].cycles;
#ifdef BARE_METAL
        unsigned long long ms = (unsigned long long)(c / ((uint64_t)CPU_MHZ * 1000ULL));
        if (!first) TIMING_LOG(", ");
        TIMING_LOG("%s %llu", op, ms);
#else
        double ms = (double)c / 1000.0;
        if (!first) TIMING_LOG(", ");
        TIMING_LOG("%s %.2f", op, ms);
#endif
        first = 0;
    }
    TIMING_LOG(" ms\n");
    s_cursor = i;
}

void yolo_timing_reset(void) {
    s_count = 0;
    s_cursor = 0;
}
