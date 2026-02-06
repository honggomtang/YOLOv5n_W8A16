/**
 * 플랫폼 설정 (BARE_METAL: Vitis DDR 맵, 호스트: 무관)
 * BARE_METAL 시 xparameters.h 로 XPAR_* 덮어쓰기 가능.
 * DDR 맵: 0x80000000 코드/스택/힙 32MB, 0x82000000 피처맵 풀 32MB,
 *         0x88000000 가중치 16MB, 0x8E000000 Detect 9MB, 0x8F000000 이미지+결과 16MB.
 */
#ifndef PLATFORM_CONFIG_H
#define PLATFORM_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef BARE_METAL

#ifdef XPAR_DDR_MEM_BASEADDR
#define PLATFORM_DDR_BASE ((uintptr_t)XPAR_DDR_MEM_BASEADDR)
#else
#define PLATFORM_DDR_BASE 0x80000000u
#endif

#ifndef WEIGHTS_DDR_BASE
#define WEIGHTS_DDR_BASE  (PLATFORM_DDR_BASE + 0x08000000u)  /* 0x88000000 */
#endif
#ifndef WEIGHTS_DDR_SIZE
#define WEIGHTS_DDR_SIZE  (16u * 1024u * 1024u)
#endif
/* W8A32: USE_WEIGHTS_W8 시 weights_w8.bin DDR 주소 (scale은 w8 내부 포함) */
#ifndef WEIGHTS_W8_DDR_BASE
#define WEIGHTS_W8_DDR_BASE  WEIGHTS_DDR_BASE
#endif
#ifndef WEIGHTS_W8_DDR_SIZE
#define WEIGHTS_W8_DDR_SIZE  (8u * 1024u * 1024u)  /* 8MB (weights_w8.bin ~2MB, 여유 확보) */
#endif

#ifndef DETECT_HEAD_BASE
#define DETECT_HEAD_BASE  (PLATFORM_DDR_BASE + 0x0E000000u)
#endif
#ifndef DETECT_HEAD_SIZE
#define DETECT_HEAD_SIZE  (9u * 1024u * 1024u)
#endif

#ifndef IMAGE_AND_FEATURE_BASE
#define IMAGE_AND_FEATURE_BASE (PLATFORM_DDR_BASE + 0x0F000000u)  /* 0x8F000000 */
#endif
#ifndef IMAGE_AND_FEATURE_SIZE
#define IMAGE_AND_FEATURE_SIZE (16u * 1024u * 1024u)  /* 16MB */
#endif

#ifndef IMAGE_DDR_BASE
#define IMAGE_DDR_BASE    IMAGE_AND_FEATURE_BASE
#endif
#define IMAGE_HEADER_SIZE 24u
#define IMAGE_DATA_SIZE   (3u * 640u * 640u * sizeof(float))
#define IMAGE_DDR_SIZE    (IMAGE_HEADER_SIZE + IMAGE_DATA_SIZE)
/* W8A16: preprocessed_image_a16.bin = 24B header + int16 Q6.10 (zero-copy L0 input) */
#define IMAGE_A16_DATA_SIZE  (3u * 640u * 640u * sizeof(int16_t))
#define IMAGE_A16_DDR_SIZE   (IMAGE_HEADER_SIZE + IMAGE_A16_DATA_SIZE)

#ifndef FEATURE_POOL_BASE
#define FEATURE_POOL_BASE (PLATFORM_DDR_BASE + 0x02000000u)
#endif
#ifndef FEATURE_POOL_SIZE
/* W8A16: 스크래치(x0, l0..l23, C3/SPPF/bottleneck 내부) 합계가 32MB 초과 → 48MB 사용. 0x82000000~0x84FFFFFF, 가중치(0x88000000) 전까지 여유 있음. */
#if defined(USE_W8A16)
#define FEATURE_POOL_SIZE (48u * 1024u * 1024u)
#else
#define FEATURE_POOL_SIZE (32u * 1024u * 1024u)
#endif
#endif

#ifndef DETECTIONS_OUT_BASE
#define DETECTIONS_OUT_BASE (IMAGE_AND_FEATURE_BASE + IMAGE_AND_FEATURE_SIZE - 4096u)
#endif

/** CPU 클럭 (MHz). mcycle → ms 변환: cycles / (CPU_MHZ * 1000) */
#ifndef CPU_MHZ
#define CPU_MHZ 100
#endif

#endif /* BARE_METAL */

#endif /* PLATFORM_CONFIG_H */
