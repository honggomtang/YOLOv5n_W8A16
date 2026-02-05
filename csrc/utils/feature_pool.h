/**
 * 피처맵 풀 할당.
 * - feature_pool_alloc/free: First-fit + coalescing. 추론 한 번에 alloc/free가 수십 번 발생하면
 *   파편화·속도 문제 가능 (MicroBlaze 등). BARE_METAL에서는 scratch 전용 사용 권장.
 * - feature_pool_scratch_*: 스크래치패드(단일 포인터 밀기). 추론 시작 시 scratch_reset(),
 *   추론 중 scratch_alloc()만 사용, free 없음. 파편화 없고 O(1).
 */
#ifndef FEATURE_POOL_H
#define FEATURE_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void feature_pool_init(void);
void* feature_pool_alloc(size_t size);
void feature_pool_free(void* ptr);
void feature_pool_reset(void);

/** 추론 시작 시 한 번 호출. 이후 scratch_alloc만 사용 시 파편화 없음. */
void feature_pool_scratch_reset(void);
/** 스크래치패드 할당 (포인터 밀기). free 없음. */
void* feature_pool_scratch_alloc(size_t size);

size_t feature_pool_get_largest_free(void);

#ifdef __cplusplus
}
#endif

#endif /* FEATURE_POOL_H */
