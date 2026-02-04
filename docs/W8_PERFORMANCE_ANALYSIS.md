# W8A32 L0 성능 분석 (2번·3번 항목)

## 2번: Data Cache - 0x88000000 캐시 대상 여부

### xparameters.h 확인 결과

| 항목 | 값 | 비고 |
|------|-----|------|
| XPAR_MIG_0_BASEADDRESS | 0x80000000 | DDR 시작 |
| XPAR_MIG_0_HIGHADDRESS | 0x8fffffff | DDR 끝 (256MB) |
| XPAR_MICROBLAZE_RISCV_USE_DCACHE | 1 | D-Cache 사용 |
| XPAR_MICROBLAZE_RISCV_DCACHE_BYTE_SIZE | 16384 (16KB) | 캐시 크기 |
| XPAR_MICROBLAZE_RISCV_DCACHE_LINE_LEN | 16 | 라인 크기 |

**WEIGHTS_W8_DDR_BASE (0x88000000)**: MIG 범위 0x80000000~0x8fffffff **안에 포함** ✓

### 결론 및 추가 확인

- xparameters.h에는 **캐시 제외 주소 범위**가 정의되어 있지 않음.
- MicroBlaze D-Cache는 보통 MIG(DDR) 전체를 캐시 대상으로 취급.
- **추가 확인 필요**: Vivado 블록 디자인에서 MicroBlaze ↔ MIG 연결
  - Cache가 MIG 앞단에 있으면 0x88000000 접근도 캐시됨.
  - MIG가 non-cacheable 포트에 붙어 있으면 캐시 미적용.
- **확인 방법**: Vivado → Block Design → MicroBlaze Data 경로 확인.
  - `microblaze_0_M_AXI_DC` → `axi_dcache` → `axi_interconnect` → `mig_7series_0` 이면 캐시 경유.

### platform_config.h 호환

- `XPAR_DDR_MEM_BASEADDR`가 xparameters.h에 **없음**.
- `platform_config.h`는 `PLATFORM_DDR_BASE = 0x80000000` 기본값 사용.
- `main.c`에서 `Xil_DCacheInvalidateRange(WEIGHTS_DDR_BASE, WEIGHTS_DDR_SIZE)` 호출 → 0x88000000 영역 무효화 후 캐시 사용.

---

## 3번: int8_t 바이트 단위 접근 비효율

### 현재 동작

```c
for (int32_t kw = 0; kw < k_w; kw++) {
    sum_xw += (*x_row++) * (float)(*w_row++);  // 1바이트씩 로드
}
```

- `*w_row++` → int8_t 1바이트 로드.
- MicroBlaze는 32비트 버스 기준이어서, 1바이트 로드 시:
  - 전체 32비트 워드 로드 후 해당 바이트 추출.
  - 또는 별도 바이트 로드 명령 사용.

### 비효율 요인

1. **메모리 트래픽**: 4개 연속 바이트를 4번 로드하면 버스 활용이 떨어짐.
2. **마스킹/시프트**: 8비트만 사용하므로 추가 연산 가능.
3. **캐시 라인**: 16바이트 단위이므로 4바이트 단위 로드가 더 유리.

### 최적화: 4바이트 정렬 시 32비트 로드

- 4바이트 정렬된 주소에서 `uint32_t` 한 번 로드 → 4개 int8 추출.
- 커널 row stride가 4의 배수가 아니면(예: 6×6) 많은 row에서 비정렬.
- **6×6 예**: row 0만 정렬, row 1~5는 정렬 아님 → row 0에서만 4바이트 로드 적용 가능.

### 적용한 최적화 (conv2d.c) — 현재

- **local_w pre-load**: (ic,b)당 1회 int8→float 변환. (dh,dw) 루프는 FP32 연산만.
- **Pre-load 단계 32비트 번들 로드**: 정렬 시 8바이트(2×uint32)씩 unroll, remainder 바이트 로드.
- **1×1 Fast Path**: kh,kw 루프 없음, ic outer로 입력 1회 로드→여러 oc 재사용.

→ 전체 구현 내역은 `W8A32_IMPLEMENTATION.md` 참고.
