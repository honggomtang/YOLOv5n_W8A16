#ifdef BARE_METAL

#include "uart_dump.h"
#include "xil_printf.h"
#include <stddef.h>
#include <stdint.h>

#define HW_DETECTION_SIZE 12

void yolo_uart_send_detections(const void* hw_detections, uint8_t count) {
    if (!hw_detections) return;
    xil_printf("YOLO\n");
    xil_printf("%02X\n", (unsigned int)count);
    const uint8_t* p = (const uint8_t*)hw_detections;
    for (uint8_t i = 0; i < count; i++) {
        for (int j = 0; j < HW_DETECTION_SIZE; j++) {
            xil_printf("%02X", (unsigned int)p[j]);
        }
    }
    xil_printf("\n");
}

#endif /* BARE_METAL */
