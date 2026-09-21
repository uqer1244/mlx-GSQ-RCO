# MLX-GSQ-RCO 포팅 진행 상황

마지막 검증: 2026-09-21

## 목표 경로

```text
GGUF mixed IQ weights
  → packed U8 safetensors + manifest
  → MLX PackedModelStore
  → quant type별 custom Metal kernel
  → Qwen3.5 hybrid runtime
```

## 검증 완료

- 원본 GGUF v3 헤더 및 866개 tensor directory 검증
- 실제 architecture가 `qwen35`임을 확인
- 27,320,697,856 parameters 및 13개 storage type 분포 확인
- 전체 텐서가 포함된 `mlx_gsq/gguf/analysis.json` 재생성
- GGUF tensor payload를 변경하지 않는 streaming safetensors converter 구현
- tensor별 shape, GGUF quant type, offset, byte count manifest 저장
- 866개 tensor에 대해 GGUF 원본과 safetensors payload SHA-256 전수 비교 통과
- 표준 `safetensors.safe_open`으로 866개 tensor 로딩 확인
- MLX `mx.load`로 866개 packed U8 tensor 로딩 확인
- F32/F16/BF16 plain tensor의 MLX dtype reinterpretation 경로 구현
- 11개 mixed quant type Metal block decoder 구현
- 각 타입별 실제 tensor 64 blocks, 총 180,224 weights가 gguf-py reference와 원소 단위 일치
- safetensors 내장 qtype metadata → decoder registry 자동 routing 검증
- 외부 manifest 없이 11개 타입 모두 safetensors → Metal decode 성공
- 11개 타입 fused Metal QMV 구현 및 dense reference matmul 비교 통과
- IQ2_S embedding 선택-row Metal decode 구현
- mlx-lm Qwen3.5 64-layer adapter 구현
- 실제 end-to-end logits `[1, 1, 248320]` 생성 성공
- mlx-lm `generate()` 2-token 실행 성공 (peak memory 10.31 GB)
- tokenizer/config assets를 `Qwen3.8-27B-GSQ-RCO/tokenizer`에 구성
- 현재 mlx-lm에 Qwen3.5 hybrid model이 존재함을 확인
- llama.cpp Metal reference와 raw `Hello` 첫-token 전체 logits 비교 통과
  - argmax 및 top-20 일치, top-100 100/100 일치
  - cosine similarity 0.9999975, centered cosine 0.9999923
  - RMSE 0.00876, max absolute error 0.04160
- 2-token recurrence 입력 `Hello,` 전체 logits 비교 통과
  - argmax 및 top-50 일치, top-100 99/100 일치
  - cosine similarity 0.9999980, centered cosine 0.9999943
  - RMSE 0.00820, max absolute error 0.03998
- GGUF `ssm_a = -exp(A_log)` 역변환 및 GDN cyclic Q/K head routing 구현
- 전체 자동화 테스트 32개 통과

## 생성 결과

```text
Qwen3.8-27B-GSQ-RCO/model-packed.safetensors
Qwen3.8-27B-GSQ-RCO/model-packed.safetensors.manifest.json
mlx_gsq/gguf/analysis.json
```

`model-packed.safetensors`는 일반 mlx-lm quantized weight가 아니다. IQ payload를
원본 그대로 담은 `mlx-gsq-packed-v1` 컨테이너이며 custom kernel이 필요하다.

## 아직 미완료

- prefill 최적화용 QMM (현재 QMV가 여러 input row를 처리하여 기능상 동작)
- QMV/QMM 성능 튜닝 및 codebook runtime dependency 제거

## 다음 correctness gate

1. 채팅 템플릿을 적용한 다양한 prompt로 multi-token generation 비교
2. prefill용 tiled QMM 구현
3. QMV reduction/vector-load 성능 튜닝
4. generation throughput 및 peak-memory benchmark

placeholder가 반환하는 기존 kernel 파일은 구현 완료로 간주하지 않는다.
