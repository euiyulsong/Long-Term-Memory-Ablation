````markdown
# LoCoMo Memory Representation / Routing 실험 결과

## 1. 실험 설정

모든 방법은 동일한 질문에 대해 BGE-M3로 **Top-5 retrieval**을 수행했고, 차이는 memory representation과 routing 방식이다.

| Method | 설명 |
|---|---|
| `raw_c2` | 연속 2-turn chunk, non-overlap |
| `raw_c6_no_overlap` | 연속 6-turn chunk, non-overlap |
| `raw_c6_overlap` | 6-turn chunk, overlap 적용 |
| `session_summary` | LoCoMo에서 제공하는 session-level summary |
| `router_raw_vs_summary` | raw 방식 중 하나를 고른 뒤, 선택된 raw와 summary 중 하나를 다시 선택 |
| `self_router_all` | C2 + C6 no-overlap + C6 overlap + summary의 Top-5를 모두 LLM에 넣고 최종 답 생성 |

평가는 50개 QA에 대해 수행했고, 주요 metric은 다음과 같다.

- `EM`: relaxed exact match
- `STRICT`: strict exact match
- `F1`: token-level F1
- `AVG_CHARS`: 최종 QA 입력 context 길이
- `Evidence Hit@5`: gold evidence가 retrieved Top-5에 포함되는 비율

---

## 2. 전체 성능

| Method | EM | STRICT EM | F1 | Avg. Context Chars |
|---|---:|---:|---:|---:|
| raw_c2 | 0.300 | 0.200 | 0.336 | 2,244 |
| raw_c6_no_overlap | 0.300 | 0.180 | 0.321 | 6,219 |
| raw_c6_overlap | 0.280 | 0.220 | 0.320 | 6,268 |
| session_summary | 0.220 | 0.080 | 0.168 | 3,822 |
| router_raw_vs_summary | 0.300 | 0.260 | 0.363 | 5,629 |
| **self_router_all** | **0.440** | **0.340** | **0.524** | **18,553** |

### 핵심 결과

가장 성능이 높은 방법은 명확하게 `self_router_all`이다.

- EM: `0.30 → 0.44`
- F1: `0.336 → 0.524`
- STRICT EM도 `0.34`로 가장 높음

즉 하나의 memory representation만 사용하는 것보다,

> 여러 representation의 retrieval 결과를 동시에 제공하고 LLM이 직접 evidence를 비교/조합하게 하는 방식

이 가장 효과적이었다.

다만 context 크기는 약 `18.5k chars`로 `raw_c2` 대비 약 **8.3배** 크다.

따라서 `self_router_all`은 성능 면에서는 가장 좋지만 비용/latency 측면에서는 가장 비싼 방식이다.

---

## 3. Chunk Size: C2 vs C6

### C2와 C6의 QA 성능

| Method | EM | F1 |
|---|---:|---:|
| raw_c2 | **0.300** | **0.336** |
| raw_c6_no_overlap | 0.300 | 0.321 |
| raw_c6_overlap | 0.280 | 0.320 |

의외로 **작은 C2 chunk가 가장 좋은 단일 raw representation**이었다.

C6는 더 많은 주변 문맥을 포함하지만, 최종 QA 성능은 개선되지 않았다.

즉:

> retrieval context가 넓다고 항상 QA 성능이 올라가지는 않는다.

C6에서는 relevant evidence 외의 주변 발화까지 많이 들어가면서 noise가 증가한 것으로 볼 수 있다.

특히 `raw_c2`는 평균 입력이 2,244 chars로 가장 작으면서도 C6와 같거나 더 좋은 성능을 냈다.

따라서 **single-representation baseline으로는 C2가 가장 효율적**이다.

---

## 4. Overlap의 효과

Retrieval 자체에서는 overlap 효과가 매우 명확했다.

| Method | Evidence Hit@5 |
|---|---:|
| raw_c2 | 0.68 |
| raw_c6_no_overlap | 0.72 |
| **raw_c6_overlap** | **0.80** |

C6 non-overlap에서 C6 overlap으로 바꾸면:

- Hit@5: `0.72 → 0.80`

즉 chunk boundary 때문에 놓치던 evidence를 overlap이 실제로 복구하고 있다.

하지만 QA 성능은:

- C6 no-overlap EM: `0.30`
- C6 overlap EM: `0.28`

로 오히려 약간 낮다.

이 결과는 중요한 포인트다.

> **Retrieval recall 증가 ≠ 최종 QA 성능 증가**

Overlap은 relevant evidence를 더 자주 포함시키지만, 동시에 비슷한 chunk가 여러 개 검색되어 redundancy와 distraction도 증가한다.

즉 overlap은 **retrieval 관점에서는 유리하지만 generation 관점에서는 항상 유리하지 않다.**

---

## 5. Session Summary 성능

`session_summary`는 가장 낮은 성능을 보였다.

| Metric | Result |
|---|---:|
| EM | 0.220 |
| STRICT | 0.080 |
| F1 | 0.168 |
| Hit@5 | 0.780 |

특이한 점은 retrieval hit는 꽤 높다는 것이다.

- summary Hit@5 = `0.78`
- C2 Hit@5 = `0.68`

그런데 QA 성능은 summary가 훨씬 낮다.

즉 summary가 **관련 session은 잘 찾아오지만**, 실제 답에 필요한 세부 정보가 summary 과정에서 사라진 경우가 많다는 의미다.

정리하면:

> Summary retrieval은 coarse-grained relevance에는 강하지만 exact factual QA에는 정보 손실이 크다.

특히 정확한 표현, 세부 attribute, list, temporal relation 등을 요구하는 질문에서는 raw dialogue가 더 유리한 것으로 보인다.

---

## 6. Explicit Router 성능

`router_raw_vs_summary` 결과:

- EM = `0.30`
- STRICT = `0.26`
- F1 = `0.363`

단일 raw보다 F1은 조금 좋아졌지만 EM에서는 거의 개선이 없다.

Router의 선택 분포를 보면 이유가 보인다.

### Raw chunk router

| Choice | Count |
|---|---:|
| raw_c2 | 6 (12%) |
| raw_c6_no_overlap | 2 (4%) |
| **raw_c6_overlap** | **42 (84%)** |

Router가 지나치게 `raw_c6_overlap`을 선호했다.

하지만 실제 단일-method 결과에서는:

- C2 EM = `0.30`
- C6 overlap EM = `0.28`

이므로 router preference와 실제 downstream 성능이 잘 맞지 않는다.

즉 router가:

> “더 넓은 context + overlap = 더 좋은 retrieval”

이라는 식의 representation-level heuristic에 과도하게 끌렸을 가능성이 있다.

실제 per-query usefulness를 충분히 판단하지 못한 것이다.

---

## 7. Raw vs Summary Router

두 번째 router의 선택은 더 극단적이다.

| Choice | Count |
|---|---:|
| **raw** | **47 (94%)** |
| summary | 3 (6%) |

이 결과는 실제 baseline 결과와 어느 정도 일치한다.

실제로:

- raw_c2 EM = `0.30`
- summary EM = `0.22`

이므로 summary보다 raw가 대부분 유리했다.

하지만 summary가 필요한 일부 query를 잘 선별했다고 보기는 어렵다.

전체적으로 explicit router는:

> 어떤 representation 하나를 선택하는 방식 자체가 성능 상한을 제한

하고 있다.

한 번 잘못 선택하면 다른 representation에 있던 evidence를 완전히 버리기 때문이다.

---

## 8. Self Router All이 강한 이유

`self_router_all`:

- EM = **0.44**
- F1 = **0.524**

으로 다른 모든 실제 inference method를 크게 앞섰다.

이 방식은 실제로 explicit router라기보다는:

> **all-context fusion + implicit evidence selection**

에 가깝다.

즉 모델이 다음을 모두 본다.

```text
C2 Top-5
+
C6 No-overlap Top-5
+
C6 Overlap Top-5
+
Session Summary Top-5
````

이 구조에서는:

1. C2의 precise evidence
2. C6의 broader context
3. overlap이 복구한 boundary evidence
4. summary의 high-level information

을 동시에 활용할 수 있다.

따라서 한 representation이 놓친 정보를 다른 representation이 보완한다.

---

## 9. Oracle 결과가 특히 중요함

Oracle 결과:

| Oracle     |    EM |    F1 |
| ---------- | ----: | ----: |
| oracle_raw | 0.440 | 0.476 |
| oracle_all | 0.480 | 0.489 |

`oracle_raw EM = 0.44`라는 것은:

> C2 / C6-no / C6-overlap 중 질문마다 가장 잘 맞는 결과를 고를 수 있다면 EM 44%까지 가능

하다는 뜻이다.

하지만 실제 router는 EM 30%밖에 못 냈다.

즉 **routing headroom이 매우 크다.**

```text
best fixed raw ≈ 0.30
actual router  ≈ 0.30
oracle raw     = 0.44
```

현재 explicit router가 거의 oracle gain을 회수하지 못하고 있다.

---

## 10. 더 흥미로운 점: Self Router가 Oracle Raw와 동일한 EM

```text
self_router_all EM = 0.44
oracle_raw      EM = 0.44
```

Self-router가 실제 inference 방식임에도 raw oracle 수준까지 올라왔다.

또 F1에서는:

```text
self_router_all = 0.524
oracle_all      = 0.489
```

으로 self-router가 oracle-all의 per-method 최대 F1보다 더 높다.

이건 모순이 아니다.

Oracle은:

```text
max(
    F1(C2),
    F1(C6-no),
    F1(C6-overlap),
    F1(summary)
)
```

만 고른다.

반면 `self_router_all`은 **여러 representation의 evidence를 합쳐 새로운 answer를 만들 수 있다.**

따라서:

> self-router의 장점은 단순히 “좋은 retrieval 하나를 고르는 것”이 아니라, 여러 retrieval의 complementary evidence를 합치는 데 있다.

이게 이번 실험에서 가장 중요한 결과 중 하나다.

---

## 11. Retrieval과 QA 사이의 관계

결과를 나란히 보면:

| Method         |    Hit@5 |       EM |
| -------------- | -------: | -------: |
| raw_c2         |     0.68 | **0.30** |
| raw_c6_no      |     0.72 | **0.30** |
| raw_c6_overlap | **0.80** |     0.28 |
| summary        |     0.78 |     0.22 |

Retrieval hit와 downstream EM의 순위가 거의 맞지 않는다.

따라서 시스템 평가를 retrieval recall만으로 해서는 안 된다.

이번 결과에서는:

```text
retrieval quality
+
context noise
+
information granularity
+
answer generation
```

이 모두 최종 성능에 영향을 준다.

특히 C6 overlap은 retrieval recall은 최고지만 QA는 C2보다 낮았다.

---

## 12. 비용 대비 성능

`self_router_all`의 가장 큰 약점은 context cost다.

| Method          | Avg chars |       EM |
| --------------- | --------: | -------: |
| raw_c2          |     2,244 |     0.30 |
| router          |     5,629 |     0.30 |
| self_router_all |    18,553 | **0.44** |

`router_raw_vs_summary`는 context를 약 2.5배 더 쓰면서도 C2 대비 EM 개선이 없다.

반면 self-router는 context를 약 8배 쓰면서 EM을 +0.14 올린다.

따라서 현재 결과만 보면 production 후보는 사실 두 개가 가장 명확하다.

```text
Cheap path:
raw_c2

High-quality path:
all_context_fusion
```

현재 explicit router는 cost/performance 관점에서 애매하다.

---

# 최종 결론

## 1. 단일 memory representation으로는 C2가 가장 좋다

2-turn chunk가 가장 작고 효율적이면서 C6보다 동등하거나 더 높은 QA 성능을 보였다.

```text
C2 EM = 0.30
C6-no EM = 0.30
C6-overlap EM = 0.28
```

따라서 기본 retrieval unit을 무조건 크게 잡을 필요는 없다.

## 2. Overlap은 retrieval recall을 개선하지만 QA 성능을 보장하지 않는다

```text
C6 no overlap Hit@5 = 0.72
C6 overlap Hit@5    = 0.80
```

하지만 EM은 개선되지 않았다.

즉 overlap은 **boundary recall 문제를 해결하지만 redundancy/noise trade-off가 존재**한다.

## 3. Session summary 단독 사용은 exact QA에 불리하다

Summary는 relevant session retrieval은 잘하지만 정보 압축 과정에서 세부 답이 소실된다.

따라서 summary는 raw memory 대체재라기보다는 **보조 representation**으로 쓰는 것이 더 적합하다.

## 4. 현재 explicit router는 잘 작동하지 않는다

Router는 `raw_c6_overlap`을 84% 선택했지만, 실제로 C6-overlap이 최고 baseline은 아니다.

또 raw를 94% 선택하면서 summary 사용 가능성도 거의 활용하지 않는다.

```text
router EM = 0.30
oracle_raw EM = 0.44
```

이 gap은 router 개선 여지가 매우 크다는 뜻이다.

## 5. 가장 좋은 방식은 all-context fusion이다

`self_router_all`이:

```text
EM     = 0.44
STRICT = 0.34
F1     = 0.524
```

로 가장 높은 성능을 보였다.

특히 F1이 oracle-all보다 높다는 점에서, 효과는 단순 representation 선택이 아니라 **서로 다른 retrieval 결과의 evidence fusion**에서 오는 것으로 보인다.

## 6. Production 관점에서는 adaptive fusion이 다음 단계다

현재 가장 합리적인 구조는:

```text
default
   ↓
RAW_C2
   ↓
confidence sufficient?
   ├─ yes → answer
   └─ no
        ↓
C2 + C6 + Summary
        ↓
all-context fusion
```

즉 항상 18k chars를 넣기보다는,

> 쉬운 질문은 C2로 처리하고, retrieval ambiguity / multi-hop / temporal / low-confidence 상황에서만 all-context fusion을 호출

하는 **conditional fusion gate**가 다음 실험으로 가장 적합하다.

```
```
