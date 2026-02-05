/**
 * 피처맵 풀: First-fit 할당자 (버퍼 재사용)
 */
#include "feature_pool.h"
#include <stddef.h>
#include <stdint.h>

#ifdef BARE_METAL
#include "platform_config.h"
#endif
#ifndef BARE_METAL
#include <stdlib.h>
#endif

#define ALIGN 8
#ifdef BARE_METAL
#define HEADER_SIZE 8u
#else
#define HEADER_SIZE (2u * (size_t)sizeof(size_t))
#endif
#define MIN_SPLIT (HEADER_SIZE * 2)
#define NIL ((size_t)-1)

static uint8_t* pool_base;
static size_t pool_size;
#ifndef BARE_METAL
static uint8_t* host_pool;
#endif

static size_t free_head;

/* 스크래치패드: 포인터 하나만 밀어서 할당. free 없음, 파편화 없음. */
static size_t scratch_offset;

static inline size_t align_up(size_t x, size_t a) {
    return (x + a - 1) & ~(a - 1);
}

void feature_pool_init(void) {
#ifdef BARE_METAL
    pool_base = (uint8_t*)FEATURE_POOL_BASE;
    pool_size = FEATURE_POOL_SIZE;
#else
    pool_size = 22u * 1024u * 1024u;  /* 호스트: 22MB (W8A32) */
#ifdef USE_W8A16
    pool_size = 48u * 1024u * 1024u;  /* W8A16 scratch: 모든 중간 텐서 연속 할당 (~24MB+ 여유) */
#endif
    host_pool = (uint8_t*)malloc(pool_size);
    pool_base = host_pool;
    if (!pool_base) pool_size = 0;
#endif
    free_head = NIL;
    scratch_offset = align_up(HEADER_SIZE, ALIGN);
    if (pool_base && pool_size >= HEADER_SIZE * 2) {
        size_t* hdr = (size_t*)(pool_base + 0);
        hdr[0] = pool_size;
        hdr[1] = NIL;
        free_head = 0;
    }
}

void* feature_pool_alloc(size_t size) {
    if (!pool_base || size == 0) return NULL;
    size_t need = align_up(size, ALIGN) + HEADER_SIZE;
    if (need > pool_size) return NULL;

    size_t prev = NIL;
    size_t curr = free_head;
    while (curr != NIL) {
        size_t* blk = (size_t*)(pool_base + curr);
        size_t blk_size = blk[0];
        size_t next = blk[1];
        if (blk_size >= need) {
            if (blk_size >= need + MIN_SPLIT) {
                size_t rest = blk_size - need;
                blk[0] = need;
                size_t* rest_blk = (size_t*)(pool_base + curr + need);
                rest_blk[0] = rest;
                rest_blk[1] = next;
                if (prev == NIL)
                    free_head = curr + need;
                else
                    ((size_t*)(pool_base + prev))[1] = curr + need;
            } else {
                if (prev == NIL)
                    free_head = next;
                else
                    ((size_t*)(pool_base + prev))[1] = next;
            }
            return (void*)(pool_base + curr + HEADER_SIZE);
        }
        prev = curr;
        curr = next;
    }
    return NULL;
}

static void unlink_free_block(size_t target, size_t prev_of_target) {
    size_t next = ((size_t*)(pool_base + target))[1];
    if (prev_of_target == NIL)
        free_head = next;
    else
        ((size_t*)(pool_base + prev_of_target))[1] = next;
}

static void insert_free_by_address(size_t curr, size_t curr_size) {
    size_t* blk = (size_t*)(pool_base + curr);
    blk[0] = curr_size;
    size_t prev_link = NIL;
    size_t w = free_head;
    while (w != NIL && w < curr) {
        prev_link = w;
        w = ((size_t*)(pool_base + w))[1];
    }
    blk[1] = w;
    if (prev_link == NIL)
        free_head = curr;
    else
        ((size_t*)(pool_base + prev_link))[1] = curr;
}

void feature_pool_free(void* ptr) {
    if (!ptr || !pool_base) return;
    uint8_t* p = (uint8_t*)ptr;
    if (p < pool_base + HEADER_SIZE || p >= pool_base + pool_size) return;
    size_t curr = (size_t)(p - pool_base - HEADER_SIZE);
    size_t* blk = (size_t*)(pool_base + curr);
    size_t curr_size = blk[0];

    insert_free_by_address(curr, curr_size);
    size_t prev_link = NIL;
    size_t w = free_head;
    while (w != NIL && w != curr) {
        prev_link = w;
        w = ((size_t*)(pool_base + w))[1];
    }
    size_t base = curr;
    size_t base_size = curr_size;
    size_t* base_blk = blk;
    if (prev_link != NIL) {
        size_t* pl = (size_t*)(pool_base + prev_link);
        if (prev_link + pl[0] == curr) {
            pl[0] += curr_size;
            unlink_free_block(curr, prev_link);
            base = prev_link;
            base_size = pl[0];
            base_blk = pl;
        }
    }

    size_t next_in_list = base_blk[1];
    if (next_in_list != NIL) {
        size_t* nl = (size_t*)(pool_base + next_in_list);
        if (base + base_size == next_in_list) {
            base_blk[0] = base_size + nl[0];
            unlink_free_block(next_in_list, base);
        }
    }
}

void feature_pool_scratch_reset(void) {
    scratch_offset = align_up(HEADER_SIZE, ALIGN);
}

void* feature_pool_scratch_alloc(size_t size) {
    if (!pool_base || size == 0) return NULL;
    size_t need = align_up(size, ALIGN);
    if (scratch_offset + need > pool_size) return NULL;
    void* ptr = (void*)(pool_base + scratch_offset);
    scratch_offset += need;
    return ptr;
}

void feature_pool_reset(void) {
#ifndef BARE_METAL
    if (host_pool) {
        free(host_pool);
        host_pool = NULL;
        pool_base = NULL;
        pool_size = 0;
        free_head = NIL;
        return;
    }
#endif
    free_head = NIL;
    if (pool_base && pool_size >= 16) {
        size_t* hdr = (size_t*)(pool_base + 0);
        hdr[0] = pool_size;
        hdr[1] = NIL;
        free_head = 0;
    }
}

size_t feature_pool_get_largest_free(void) {
    size_t max_free = 0;
    if (!pool_base) return 0;
    size_t curr = free_head;
    while (curr != NIL) {
        size_t* blk = (size_t*)(pool_base + curr);
        size_t blk_size = blk[0];
        if (blk_size > max_free) max_free = blk_size;
        curr = blk[1];
    }
    return max_free;
}
