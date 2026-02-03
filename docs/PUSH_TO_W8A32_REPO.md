# YOLOv5n_W8A32 저장소로 푸시하기

다른 저장소(https://github.com/honggomtang/YOLOv5n_W8A32)에 지금까지 작업한 W8A32 코드를 올리는 방법입니다.

## 1. GitHub에서 새 저장소 생성

- https://github.com/new 에서 **YOLOv5n_W8A32** 이름으로 저장소 생성
- Public/Private 선택 후 생성 (README 추가 여부는 상관없음)

## 2. 원격 추가 및 푸시 (프로젝트 루트에서 실행)

```bash
cd /Users/kinghong/Desktop/YOLOv5n_in_C-1

# 새 원격 추가 (한 번만)
git remote add w8a32 https://github.com/honggomtang/YOLOv5n_W8A32.git

# W8A32 관련 파일만 스테이징 (바이너리·출력물 제외)
git add README.md csrc/ docs/W8A32_PLAN.md docs/PUSH_TO_W8A32_REPO.md
git add requirements.txt run_compare_host.sh .gitignore
git add tools/quantize_weights.py tools/compare_fp32_w8.py tools/decode_detections.py
git add tests/test_c3.c tests/test_detect.c

# weights_w8.bin 포함하려면 (용량 큼, 선택)
# git add assets/weights_w8.bin

# 커밋
git commit -m "W8A32 mixed-precision: INT8 weights, FP32 compute, buffer pool, conv2d_w8, c3/detect W8 path"

# 새 저장소로 푸시 (처음이면 -u로 upstream 설정)
git push -u w8a32 main
```

이후부터는 `git push w8a32 main` 만 하면 됩니다.

## 3. 기존 origin 유지

- **origin** = 기존 YOLOv5n_in_C (변경 없음)
- **w8a32** = 새 YOLOv5n_W8A32 저장소

양쪽에 따로 푸시하려면:

- 기존 저장소: `git push origin main`
- W8A32 저장소: `git push w8a32 main`

## 4. 주의

- `main`, `tests/test_c3`, `tests/test_detect`(실행 파일), `data/output/*` 결과물은 커밋하지 않도록 되어 있습니다.
- `assets/weights_w8.bin`은 용량이 크면 나중에 추가하거나 Git LFS를 고려하세요.
