# Pointwise Self-Router 실험 결과 분석

## 1. 실험 요약

이번 실험의 `self-router`는 이전 `self_router_all`처럼 모든 문서를 한 번에 넣는 방식이 아니라,

> **검색된 각 문서를 LLM이 pointwise로 `relevant=1 / irrelevant=0` 평가한 뒤, relevant 문서만 최종 답변 생성에 사용하는 방식**

이다.

비교한 방법은 다음 4개다.

| Method | 방식 |
|---|---|
| `c6_overlap_selfrouter` | C6 overlap Top-5 → 각 문서 0/1 filtering → relevant만 QA |
| `summary_selfrouter` | Session summary Top-5 → 0/1 filtering → relevant만 QA |
| `taskrouter_selfrouter` | Query를 보고 raw/summary 선택 → Top-5 → 0/1 filtering → QA |
| `both_selfrouter` | C6 overlap Top-5 + summary Top-5 총 10개 → 각각 0/1 filtering → QA |

---

## 2. 전체 결과

| Method | EM | STRICT EM | F1 | Avg. Kept Docs | Avg. Context Chars |
|---|---:|---:|---:|---:|---:|
| c6_overlap_selfrouter | 0.200 | 0.120 | 0.209 | 1.72 | 2,121 |
| summary_selfrouter | **0.240** | 0.080 | 0.155 | **1.18** | **885** |
| taskrouter_selfrouter | 0.220 | 0.140 | 0.211 | 1.62 | 1,902 |
| **both_selfrouter** | **0.240** | **0.160** | **0.258** | 2.86 | 2,973 |

### 핵심 결과

가장 좋은 방법은 `both_selfrouter`다.

- EM = **0.24**
- STRICT EM = **0.16**
- F1 = **0.2583**

특히 F1 기준으로 다른 방식보다 확실히 높다.

즉,

> **raw와 summary를 둘 다 검색한 뒤 pointwise relevance filtering을 하는 방식이, 하나의 representation만 filtering하는 것보다 낫다.**

다만 절대 성능 자체는 이전의 `self_router_all`보다 낮다.

---

# 3. 이전 All-Context Fusion 결과와 비교

이전 실험의 `self_router_all` 결과는:

| Method | EM | STRICT | F1 | Avg. chars |
|---|---:|---:|---:|---:|
| self_router_all | **0.440** | **0.340** | **0.524** | 18,553 |

이번 최고 결과:

| Method | EM | STRICT | F1 | Avg. chars |
|---|---:|---:|---:|---:|
| both_selfrouter | 0.240 | 0.160 | 0.258 | 2,973 |

차이가 상당히 크다.

```text
EM
all-context fusion       0.44
pointwise both-router    0.24

F1
all-context fusion       0.524
pointwise both-router    0.258
````

따라서 이번 실험에서 가장 중요한 결론은:

> **검색 결과를 사전에 hard filtering하는 것이 오히려 필요한 evidence를 버린다.**

라고 볼 수 있다.

---

# 4. 왜 Pointwise Self-Router가 성능을 떨어뜨렸는가?

## 4.1 Hard decision의 information loss

Pointwise router는 각 문서를 독립적으로:

```text
relevant = 1
irrelevant = 0
```

으로 결정한다.

문제는 실제 multi-hop / temporal QA에서는 하나의 문서만 봤을 때는 relevance가 약해 보여도, 다른 문서와 같이 봤을 때 중요한 경우가 있다는 것이다.

예를 들어:

```text
Doc A:
John started a company.

Doc B:
Two months later, he sold it.

Question:
What did John do two months after starting the company?
```

Doc B만 보면 `starting the company`와 직접적으로 연결되지 않아 relevance가 애매할 수 있다.

하지만 A+B를 같이 보면 답을 구성할 수 있다.

Pointwise filtering은 이런 **cross-document dependency**를 잘 처리하지 못한다.

---

## 4.2 Relevance와 Answer Utility는 다르다

Retriever → self-router가 판단하는 것은 사실상:

> "이 문서가 질문과 관련 있는가?"

인데 최종적으로 필요한 것은:

> "이 문서가 다른 evidence들과 결합됐을 때 답을 만드는 데 도움이 되는가?"

이다.

둘은 다르다.

즉:

```text
relevance(document, query)
```

만 판단하는 것보다,

```text
utility(document | other documents, query)
```

가 실제 QA에는 더 중요하다.

이번 결과는 이 차이를 잘 보여준다.

---

# 5. C6 Pointwise Filtering 결과

`c6_overlap_selfrouter`:

```text
EM     = 0.20
F1     = 0.209
KEPT   = 1.72 / 5
```

평균적으로 Top-5 중 **약 34%만 남겼다.**

```text
1.72 / 5 = 34.4%
```

즉 꽤 공격적인 filtering이다.

이전 C6 overlap baseline은:

```text
raw_c6_overlap
EM = 0.28
F1 = 0.320
```

이었는데 pointwise filtering 후:

```text
EM = 0.20
F1 = 0.209
```

로 오히려 크게 떨어졌다.

### 해석

```text
Top-5 그대로 QA
        >
0/1 relevance filtering → QA
```

이다.

즉 BGE Top-5 안에 이미 유용한 supplementary evidence가 있었는데 self-router가 일부를 제거하면서 성능을 잃었다는 뜻이다.

---

# 6. Summary Self-Router는 조금 특이함

`summary_selfrouter`:

```text
EM     = 0.24
F1     = 0.155
KEPT   = 1.18
chars  = 885
```

매우 적은 context만 사용한다.

평균적으로:

```text
Top-5 summaries
→ 약 1.18개만 유지
```

했다.

EM은 0.24로 꽤 괜찮지만 F1은 매우 낮다.

이건 summary 특성상:

> 맞으면 짧게 정확히 맞고, 틀리면 필요한 detail 자체가 없는 경우

가 많을 가능성이 높다.

따라서 summary + pointwise filtering은 **극단적으로 저비용 memory path**로는 의미가 있지만, 최종 품질은 낮다.

---

# 7. Task Router는 큰 효과가 없었다

Task router:

```text
raw     45 / 50 = 90%
summary  5 / 50 = 10%
```

즉 거의 모든 query를 raw로 보냈다.

결과:

```text
taskrouter_selfrouter
EM = 0.22
F1 = 0.211
```

C6 self-router:

```text
EM = 0.20
F1 = 0.209
```

와 거의 차이가 없다.

### 이유

Router가 90%를 raw로 보내기 때문에 실질적으로:

```text
taskrouter_selfrouter
≈ c6_overlap_selfrouter
```

가 되어버렸다.

즉 query만 보고:

```text
raw vs summary
```

를 결정하는 coarse task routing은 큰 value를 만들지 못했다.

---

# 8. Both Self-Router가 Pointwise 방식 중에서는 가장 좋음

`both_selfrouter`는:

```text
C6 Top-5
+
Summary Top-5
= 10 docs

↓ pointwise 0/1

평균 2.86 docs 유지
```

했다.

즉 약:

```text
2.86 / 10 = 28.6%
```

만 최종 QA에 사용했다.

그런데도:

```text
EM = 0.24
F1 = 0.258
```

로 다른 pointwise 방법보다 가장 높았다.

### 의미

Raw와 summary는 서로 complementary하다.

한쪽 representation만 먼저 선택하는 것보다:

```text
raw candidates + summary candidates
→ relevance filtering
```

이 낫다.

이는 앞선 실험과도 일관된다.

> **representation을 먼저 버리는 것보다 여러 representation을 유지하는 것이 좋다.**

---

# 9. 하지만 Both Pointwise조차 All-Context Fusion에는 크게 밀림

성능과 context를 같이 비교하면:

| Method                      |       EM |        F1 | Avg chars |
| --------------------------- | -------: | --------: | --------: |
| raw_c2 baseline             |     0.30 |     0.336 |     2,244 |
| previous all-context fusion | **0.44** | **0.524** |    18,553 |
| c6 pointwise                |     0.20 |     0.209 |     2,121 |
| summary pointwise           |     0.24 |     0.155 |   **885** |
| taskrouter pointwise        |     0.22 |     0.211 |     1,902 |
| both pointwise              |     0.24 |     0.258 |     2,973 |

놀랍게도 `both_selfrouter`는 context를 약 3k까지 줄였지만 **raw_c2 baseline보다도 낮다.**

즉 현재 pointwise binary filter는:

```text
context compression은 성공
QA preservation은 실패
```

했다고 보는 게 정확하다.

---

# 10. 0/1 Binary Relevance가 너무 aggressive한 이유

이번 self-router가 이런 판단을 한다고 하자.

```text
Doc A → 1
Doc B → 0
Doc C → 0
Doc D → 1
Doc E → 0
```

그러면 B/C/E는 완전히 제거된다.

하지만 LLM answering에서는 weakly relevant evidence도 도움이 될 수 있다.

예:

```text
direct evidence       → 매우 중요
temporal support      → 중요
entity disambiguation → 중요
background context    → 때로 중요
```

Pointwise binary classifier는 이걸 전부:

```text
1 or 0
```

으로 압축한다.

따라서 information bottleneck이 너무 강하다.

---

# 11. 이번 세 실험을 종합하면

지금까지 결과를 연결하면 꽤 명확한 패턴이 나온다.

### Representation 하나만 사용

```text
C2
EM = 0.30
```

### Explicit routing으로 하나 선택

```text
router_raw_vs_summary
EM = 0.30
```

개선 없음.

### Pointwise filtering

```text
both_selfrouter
EM = 0.24
```

오히려 하락.

### 모든 representation 제공

```text
all-context fusion
EM = 0.44
```

가장 좋음.

따라서 현재 데이터에서는:

```text
hard selection
       <
hard pointwise filtering
       <
retain evidence + LLM fusion
```

이라기보다는 정확히:

```text
hard filtering / hard routing
        ↓
information loss
        ↓
성능 저하

all-context fusion
        ↓
information preservation
        ↓
LLM이 contextual하게 evidence 선택
        ↓
최고 성능
```

패턴이 나타난다.

---

# 12. 가장 중요한 결론

## Pointwise relevance filtering은 이 task에서 적합하지 않다

현재 self-router를:

> 각 document를 독립적으로 relevant / irrelevant 판단

으로 정의하면 성능이 오히려 떨어졌다.

특히:

```text
C6 baseline EM           0.28
C6 + pointwise router    0.20
```

으로 filtering 자체가 damage를 만들었다.

따라서 conversational long-term memory QA에서는 **document-level relevance가 독립적이지 않을 가능성이 크다.**

---

## Raw + Summary의 diversity 자체는 가치가 있다

Pointwise 방식 중에서는 `both_selfrouter`가 가장 좋았다.

이는:

> raw dialogue와 summary가 complementary evidence를 제공한다

는 기존 결과를 다시 지지한다.

다만 그 evidence를 binary filter로 버리기보다는 **LLM에게 같이 보여주는 것이 훨씬 낫다.**

---

## Task Router도 필요성이 낮아 보인다

Task router가:

```text
raw 90%
summary 10%
```

으로 편향되어 있고 실제 performance gain도 거의 없다.

따라서:

> query → raw/summary 선택

같은 coarse routing보다는 retrieval 이후의 evidence handling이 더 중요한 문제로 보인다.

---

# 최종 결론

이번 실험까지 종합하면 가장 좋은 구조는 현재로서는:

```text
Query
  ↓
BGE retrieval from multiple representations
  ↓
C2 / C6 / Summary candidates 유지
  ↓
LLM에게 함께 제공
  ↓
LLM이 context jointly 보고 evidence 선택·조합
  ↓
Answer
```

이다.

반대로 다음 두 방식은 현재 결과상 권장하기 어렵다.

```text
Query
→ representation 하나 hard routing
```

및

```text
Query
→ docs
→ 각 doc 독립적으로 0/1
→ irrelevant drop
```

둘 다 필요한 정보를 너무 일찍 버릴 가능성이 있다.

## 한 줄 결론

> **이 LoCoMo 실험에서는 self-routing의 핵심이 “irrelevant document를 미리 제거하는 것”이 아니라, 여러 retrieval representation을 보존한 상태에서 LLM이 jointly evidence를 선택하고 결합하도록 하는 데 있었다.**

그리고 비용까지 고려하면 다음 실험은 **hard 0/1 filtering보다는 soft filtering / top-N reranking**이 가장 적절하다.

예를 들어:

```text
10 docs
→ LLM relevance score 0~2
→ 상위 5개 유지
→ QA
```

처럼 최소 evidence budget을 보장하는 방식이면, 이번 pointwise filtering의 비용 절감 장점과 all-context fusion의 정보 보존 장점을 동시에 가져갈 수 있다.

```
```
