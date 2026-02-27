# GitHub 업로드 가이드

## 업로드 대상 (선택적)

### 필수 (csrc, vsrc)
```
csrc/          # C 소스 전체 (main, blocks, operations, utils, drivers)
vsrc/          # Conv 가속기 RTL 전체
```

### 필요한 도구·스크립트
```
tools/         # Python 도구 (export_weights_to_bin.py, export_acc_repack_from_w8.py, preprocess_*, gen_silu_lut.py 등)
build_host.bat
run_compare_host.sh
regenerate_weights_bin.bat
build_run_compare.bat
run_build.cmd
run_compare_host.sh
requirements.txt
```

### 테스트
```
tests/         # 단위 테스트 (test_*.c, test_vectors_*.h)
```

### 설정·문서
```
README.md
CHANGELOG.md
.gitignore
.gitattributes
```

### 제외 (docs)
- `docs/` 폴더는 .gitignore에 포함되어 업로드되지 않음.

### 제외 (대용량/생성물)
- `assets/*.bin`, `assets/*.pt` (.gitignore)
- `data/output/*` (생성물)
- `*.elf`, `main`, `main.exe` (빌드 결과)

---

## Git 명령 예시

```bash
# 1. docs가 이전에 추적 중이었다면 인덱스에서 제거
git rm -r --cached docs/ 2>nul || true

# 2. 업로드할 파일만 스테이징 (선택적)
git add csrc/
git add vsrc/
git add tools/
git add tests/
git add build_host.bat build_run_compare.bat regenerate_weights_bin.bat
git add run_compare_host.sh run_build.cmd
git add requirements.txt README.md CHANGELOG.md .gitignore .gitattributes

# 3. data 구조만 필요하면 (output 제외)
git add data/image/ data/input/ 2>nul || true

# 4. 커밋 및 푸시
git commit -m "Add vsrc RTL, csrc drivers, Conv accelerator support"
git push origin main
```
