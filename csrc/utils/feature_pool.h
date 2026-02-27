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

void feature_pool_scratch_reset(void);
void* feature_pool_scratch_alloc(size_t size);

size_t feature_pool_get_largest_free(void);

#ifdef __cplusplus
}
#endif

#endif /* FEATURE_POOL_H */
