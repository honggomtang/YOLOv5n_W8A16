#ifndef UART_DUMP_H
#define UART_DUMP_H

#include <stdint.h>

struct hw_detection_t;

void yolo_uart_send_detections(const void* hw_detections, uint8_t count);

#endif /* UART_DUMP_H */
