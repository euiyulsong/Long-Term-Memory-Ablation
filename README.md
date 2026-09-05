# LongMemEval Chunked Memory Ablation 결과

## 실험 설정

- Dataset: `LongMemEval oracle`
- 평가 샘플: 50개
- Question type별 균등 샘플링
- Memory / QA model: `qwen/qwen3.5-35b-a3b`
- Chunk 수: `2`
- Memory extraction max tokens: `1000`
- QA max tokens: `16`
- Workers: `50`
- Reasoning: OFF
- Structured JSON: ON
- 평가 지표: Exact Match (EM)
- 총 조건 수: 150
  - 50 examples × 3 methods
- 실패: **0 / 150**

## 비교 방법

| Method | 설명 |
|---|---|
| `untyped` | episodic/fact 구분 없이 하나의 memory list로 추출 |
| `joint_typed` | 한 번의 extraction에서 episodic memory와 fact memory를 동시에 분리 |
| `sequential_typed` | episodic extraction과 fact extraction을 각각 별도 LLM call로 수행한 뒤 결합 |

## 전체 결과

| Method | EM | 평균 Memory 수 | 평균 Extraction Calls | 평균 시간 |
|---|---:|---:|---:|---:|
| **Untyped** | **0.400** | 13.4 | 1.5 | **8.16s** |
| Joint Typed | 0.380 | 19.0 | 1.5 | 10.63s |
| Sequential Typed | 0.340 | **21.5** | 2.9 | 18.96s |

## 핵심 결과

### 1. Untyped memory가 가장 높은 정확도

가장 단순한 `untyped` 방식이 **EM 0.400**으로 가장 높았다.

성능 순서는 다음과 같다.

`Untyped (0.40) > Joint Typed (0.38) > Sequential Typed (0.34)`

Typed memory를 사용한다고 LongMemEval QA 성능이 자동으로 좋아지지는 않았다.

Joint typed는 untyped 대비:

- 절대 EM: `-0.02`
- 상대적으로 약 `5%` 감소

Sequential typed는 untyped 대비:

- 절대 EM: `-0.06`
- 상대적으로 약 `15%` 감소

---

### 2. Memory를 더 많이 추출한다고 성능이 올라가지 않았다

Memory 수는 반대로 증가했다.

| Method | Memory 수 | EM |
|---|---:|---:|
| Untyped | 13.4 | **0.400** |
| Joint Typed | 19.0 | 0.380 |
| Sequential Typed | **21.5** | 0.340 |

즉 이번 실험에서는

> **더 많은 memory ≠ 더 좋은 QA 성능**

이었다.

특히 Sequential Typed는 Untyped보다 평균 약 **60% 더 많은 memory**를 저장했지만 EM은 오히려 `0.40 → 0.34`로 떨어졌다.

이는 memory extraction에서 recall만 높이는 것보다 **관련성 높은 정보만 압축해서 유지하는 것**이 중요할 가능성을 보여준다.

---

### 3. Sequential extraction의 이점은 관찰되지 않음

Sequential Typed는 episodic과 fact를 독립적으로 추출하기 때문에 가장 많은 정보를 보존했다.

하지만 결과는:

`Joint Typed: 0.380`  
`Sequential Typed: 0.340`

으로 오히려 Joint 방식보다 낮았다.

가능한 이유는 episodic/fact를 독립적으로 추출하면서 동일하거나 유사한 정보가 중복 저장되고, 최종 QA 단계에서 irrelevant memory가 증가했기 때문일 수 있다.

즉,

> episodic/fact extraction을 서로 독립적으로 잘하는 것과 최종 QA에 좋은 memory representation을 만드는 것은 동일한 문제가 아니다.

---

### 4. 비용 측면에서도 Untyped가 가장 유리

Sequential Typed는 실제로 두 extraction 단계를 수행하기 때문에 비용이 크게 증가했다.

| Method | Extraction calls | 평균 Latency |
|---|---:|---:|
| Untyped | 1.5 | **8.16s** |
| Joint Typed | 1.5 | 10.63s |
| Sequential Typed | **2.9** | **18.96s** |

Sequential Typed는 Untyped 대비 약:

- `1.93×` extraction calls
- `2.32×` latency

를 사용했지만 성능은 더 낮았다.

따라서 현재 구성에서는 Sequential Typed를 사용할 근거가 약하다.

---

## JSON 안정성 확인

이전 구현에서는 빈 LLM response / JSON parsing failure가 다수 발생했지만 수정 후에는:

- Attempt: `150`
- Success: `150`
- Failure: **0**

으로 모든 조건이 정상 평가되었다.

따라서 이번 결과의 method 간 차이는 이전처럼 JSON parsing failure나 조건별 sample 누락 때문에 발생한 것은 아니다.

---

## 결론

이번 50-example LongMemEval 실험에서는 **단순 Untyped memory extraction이 가장 좋은 trade-off**를 보였다.

### 성능

`Untyped > Joint Typed > Sequential Typed`

### Memory 크기

`Sequential > Joint > Untyped`

### 비용

`Sequential > Joint > Untyped`

즉 현재 결과만 보면 가장 단순한 Untyped 방식이:

- 가장 높은 EM
- 가장 적은 memory
- 가장 낮은 latency

를 동시에 달성했다.

이는 memory representation을 세분화하거나 extraction 단계를 늘리는 것이 항상 도움이 되는 것은 아니며, 오히려 **memory verbosity / duplication / irrelevant information 증가가 downstream QA를 방해할 수 있음**을 시사한다.

## 다음 실험

가장 중요한 후속 실험은 `untyped`와 `joint_typed`의 차이가 실제로 유의미한지 확인하는 것이다.

현재 차이는:

`0.40 vs 0.38`

로 50개 샘플에서는 단 1문제 차이일 수 있기 때문에 통계적으로 강한 결론을 내리기는 어렵다.

따라서 다음에는 최소 500개 전체 LongMemEval에서:

- paired EM 비교
- `untyped_only`
- `joint_only`
- `both_correct`
- `both_wrong`
- question type별 EM
- memory count별 EM
- episodic/fact memory 수와 정확도의 상관관계

를 함께 보는 것이 좋다.

반면 `sequential_typed`는 현재 `0.34`로 성능과 비용 모두 열세라, 추가 최적화가 없다면 우선순위가 낮다.
