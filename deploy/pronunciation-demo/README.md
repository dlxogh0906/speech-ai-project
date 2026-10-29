# 한국어 통합 발음평가 앱

**NPU 운영에 사용하는 앱 계열이다.** 마지막 원격 확인(2026-10-25 KST)의 모델은 **M3 `w2vbert_synth_tense_v2/step_3000`**, 서비스는 `pron-api` · 포트 `8932` · NPU 7이다. 문서 갱신은 2026-10-29 KST이며, 오늘 서버를 재확인한 것은 아니다.

원격 `/root/pron_serve/run_8932.sh`가 `/root/pron_serve/ft3000_2026-10-19/` 아래 HF 모델과 `w2vbert_ctc_f192_448_1024.rbln`을 지정한다. 주요 표시 인자는 `--canon-fix --del-score-max 50 --liquid-rule`이다. 저장소 소스의 기본값이나 연구 체크포인트를 운영 모델 식별자로 사용하지 않는다.

- [현재 서비스 모델·배포 이력](../../docs/발음평가_현재_서비스_모델.md)
- [NPU API 요청 형식·인증](../../docs/NPU_서버_API_안내.md): `/sent.eval.kr`, `/speak`, `/word.judge`, `/health`
- [10월 21일 운영 반영 기록](../../docs/발음평가_개선_반영_2026-10-21.md): 모델 경로·실행 인자·검증
- [학습 계보와 데이터](../../docs/발음평가_모델별_데이터셋_학습명세_2026-10-26.md)

`deploy/serve.sh mdd`와 `deploy/infer_mdd.sh`는 기존 XLS-R `deploy/mdd-api` 경로를 실행한다. 이 앱의 NPU 운영 설정과는 별개다. H0·P3·λ=.3 파일럿의 오프라인 결과는 운영 채택을 뜻하지 않는다.
