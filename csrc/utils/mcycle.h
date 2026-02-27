#ifndef MCYCLE_H
#define MCYCLE_H

#include <stdint.h>

#if defined(BARE_METAL)

#define RISCV_CSR_MCYCLE   0xB00
#define RISCV_CSR_MCYCLEH  0xB80

static inline uint32_t mcycle_read_lo(void) {
    uint32_t v;
    __asm__ __volatile__("csrr %0, 0xB00" : "=r"(v) : : "memory");
    return v;
}

static inline uint32_t mcycle_read_hi(void) {
    uint32_t v;
    __asm__ __volatile__("csrr %0, 0xB80" : "=r"(v) : : "memory");
    return v;
}

static inline uint64_t mcycle_read64(void) {
    uint32_t hi  = mcycle_read_hi();
    uint32_t lo  = mcycle_read_lo();
    uint32_t hi2 = mcycle_read_hi();
    if (hi != hi2)
        lo = mcycle_read_lo();
    return ((uint64_t)hi2 << 32) | (uint64_t)lo;
}

static inline uint64_t mcycle_delta64(uint64_t start, uint64_t end) {
    return end - start;
}

#define timer_read64()   mcycle_read64()
#define timer_delta64(s, e)  mcycle_delta64(s, e)

#else

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
static inline uint64_t host_time_us(void) {
    static LARGE_INTEGER freq = { 0 };
    LARGE_INTEGER c;
    if (freq.QuadPart == 0)
        QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (uint64_t)((double)c.QuadPart * 1000000.0 / (double)freq.QuadPart);
}
#else
#include <sys/time.h>
static inline uint64_t host_time_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}
#endif

static inline uint64_t timer_delta64(uint64_t start, uint64_t end) {
    return end - start;
}

#define timer_read64()   host_time_us()

#endif /* BARE_METAL */

#endif /* MCYCLE_H */
