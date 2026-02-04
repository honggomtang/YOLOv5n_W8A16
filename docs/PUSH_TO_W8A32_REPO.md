# GitHub에 올리기 (가이드)

## 커밋에 포함하지 않는 것 (권장)

- `assets/weights_w8.bin` (용량 큼 → 필요하면 Git LFS 고려)
- `data/output/detections.*` (실행 결과물)

## 원격 추가 및 푸시 (예시)

프로젝트 루트에서:

```bash
git remote add w8a32 <YOUR_REPO_URL>
git push -u w8a32 main
```

이미 원격이 있다면:

```bash
git push w8a32 main
```

