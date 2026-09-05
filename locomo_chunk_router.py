#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoCoMo memory chunk / summary / routing experiment.

LOCAL
=====
BAAI/bge-m3 only

OPENROUTER
==========
QA
Router
Self-router

METHODS
=======
1. raw_c2
   chunk size=2, stride=2
   BGE TOP-5

2. raw_c6_no_overlap
   chunk size=6, stride=6
   BGE TOP-5

3. raw_c6_overlap
   chunk size=6, stride=3 default
   BGE TOP-5

4. session_summary
   LoCoMo existing session summaries
   BGE TOP-5

5. router_raw_vs_summary
   Stage 1:
      choose best raw:
        raw_c2
        raw_c6_no_overlap
        raw_c6_overlap

   Stage 2:
      choose selected raw vs session_summary

   Final QA gets ONLY selected context.

6. self_router_all
   Concatenate:
      raw_c2 TOP-5
      raw_c6_no_overlap TOP-5
      raw_c6_overlap TOP-5
      session_summary TOP-5

   One OpenRouter answer call.

METRICS
=======
strict_em:
    normalized prediction must exactly equal normalized gold

em:
    relaxed exact-match
    - exact match
    - gold appears as complete normalized phrase inside prediction
    - prediction appears as complete normalized phrase inside gold

f1:
    token F1

The QA prompt strongly instructs the model to output a minimal answer span,
so strict EM should also become substantially more meaningful.
"""

import os
import re
import csv
import json
import time
import random
import argparse
import threading

from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests


# =============================================================================
# CONFIG
# =============================================================================

LOCOMO_URL = (
    "https://raw.githubusercontent.com/"
    "snap-research/locomo/main/data/locomo10.json"
)

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)

DEFAULT_BGE_MODEL = "BAAI/bge-m3"

DEFAULT_QA_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_ROUTER_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_SELF_ROUTER_MODEL = "qwen/qwen3.5-35b-a3b"


RAW_C2 = "raw_c2"
RAW_C6_NO = "raw_c6_no_overlap"
RAW_C6_OV = "raw_c6_overlap"
SUMMARY = "session_summary"
ROUTER = "router_raw_vs_summary"
SELF_ALL = "self_router_all"


BASE_METHODS = [
    RAW_C2,
    RAW_C6_NO,
    RAW_C6_OV,
    SUMMARY,
]

RAW_METHODS = [
    RAW_C2,
    RAW_C6_NO,
    RAW_C6_OV,
]

METHODS = [
    RAW_C2,
    RAW_C6_NO,
    RAW_C6_OV,
    SUMMARY,
    ROUTER,
    SELF_ALL,
]


# =============================================================================
# UTILS
# =============================================================================

def ensure_dir(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True,
    )


def safe_mean(xs):
    return (
        sum(xs) / len(xs)
        if xs
        else 0.0
    )


def shorten(text, n=130):
    text = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    if len(text) <= n:
        return text

    return text[:n] + "..."


# =============================================================================
# METRICS
# =============================================================================

def normalize_answer(text):
    """
    SQuAD-ish normalization.
    """

    text = str(text).lower()

    # remove English articles
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    # punctuation
    text = re.sub(
        r"[^\w\s가-힣]",
        " ",
        text,
        flags=re.UNICODE,
    )

    return " ".join(
        text.split()
    )


def strict_exact_match(pred, gold):
    """
    Real strict normalized EM.
    """

    return float(
        normalize_answer(pred)
        ==
        normalize_answer(gold)
    )


def relaxed_exact_match(pred, gold):
    """
    EM-friendly semantic span match.

    Examples:

    gold:
      Contemporary

    pred:
      Contemporary

    => 1


    gold:
      Contemporary

    pred:
      Jon's favorite style of painting is contemporary.

    => 1


    gold:
      tracking inventory, resources, and donations

    pred:
      tracking inventory, resources, and donations

    => 1


    pred:
      The system helped with tracking inventory,
      resources, and donations.

    => 1
    """

    p = normalize_answer(pred)
    g = normalize_answer(gold)

    if not p or not g:
        return 0.0

    if p == g:
        return 1.0

    if g in p:
        return 1.0

    if p in g:
        return 1.0

    return 0.0


def token_f1(pred, gold):

    p = (
        normalize_answer(pred)
        .split()
    )

    g = (
        normalize_answer(gold)
        .split()
    )

    if not p and not g:
        return 1.0

    if not p or not g:
        return 0.0

    pc = Counter(p)
    gc = Counter(g)

    common = sum(
        (pc & gc).values()
    )

    if common == 0:
        return 0.0

    precision = (
        common / len(p)
    )

    recall = (
        common / len(g)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


# =============================================================================
# DATA
# =============================================================================

def download_locomo(path):

    path = Path(path)

    if path.exists():

        print(
            f"[dataset exists] {path}"
        )

        return

    print("=" * 100)
    print("DOWNLOAD LOCOMO")
    print("=" * 100)

    response = requests.get(
        LOCOMO_URL,
        timeout=120,
    )

    response.raise_for_status()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        response.content
    )

    print(
        f"[saved] {path}"
    )


def load_locomo(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    print(
        f"[LoCoMo] conversations="
        f"{len(data)}"
    )

    return data


# =============================================================================
# LOCOMO PARSING
# =============================================================================

def get_session_numbers(conv):

    result = []

    for key in conv.keys():

        m = re.fullmatch(
            r"session_(\d+)",
            key,
        )

        if m:

            result.append(
                int(
                    m.group(1)
                )
            )

    return sorted(result)


def parse_turn(turn):

    speaker = (
        turn.get("speaker")
        or turn.get("role")
        or "UNKNOWN"
    )

    text = (
        turn.get("text")
        or turn.get("content")
        or ""
    )

    caption = (
        turn.get("blip_caption")
        or ""
    )

    if caption:

        text = (
            f"{text} "
            f"[Image: {caption}]"
        ).strip()

    dia_id = (
        turn.get("dia_id")
        or ""
    )

    return {
        "text":
            f"{speaker}: {text}",

        "dia_id":
            dia_id,
    }


def flatten_conversation(sample):

    conv = sample[
        "conversation"
    ]

    turns = []

    for session_num in (
        get_session_numbers(conv)
    ):

        session_id = (
            f"session_{session_num}"
        )

        date = conv.get(
            f"session_{session_num}_date_time"
        )

        for turn in (
            conv[session_id]
        ):

            parsed = parse_turn(
                turn
            )

            turns.append(
                {
                    "session_num":
                        session_num,

                    "session_id":
                        session_id,

                    "date":
                        date,

                    "dia_id":
                        parsed[
                            "dia_id"
                        ],

                    "text":
                        parsed[
                            "text"
                        ],
                }
            )

    return turns


# =============================================================================
# CHUNKING
# =============================================================================

def make_chunks(
    turns,
    chunk_size,
    stride,
):

    docs = []

    start = 0
    chunk_index = 0

    while start < len(turns):

        sub = turns[
            start:
            start + chunk_size
        ]

        if not sub:
            break

        lines = []

        for turn in sub:

            metadata = []

            if turn["date"]:
                metadata.append(
                    str(turn["date"])
                )

            metadata.append(
                turn["session_id"]
            )

            prefix = (
                "["
                + " | ".join(metadata)
                + "]"
            )

            lines.append(
                f"{prefix} "
                f"{turn['text']}"
            )

        docs.append(
            {
                "id":
                    f"chunk_{chunk_index}",

                "text":
                    "\n".join(lines),

                "dia_ids": [
                    t["dia_id"]
                    for t in sub
                    if t["dia_id"]
                ],

                "session_ids":
                    sorted(
                        set(
                            t["session_id"]
                            for t in sub
                        )
                    ),

                "start_idx":
                    start,

                "end_idx":
                    (
                        start
                        + len(sub)
                        - 1
                    ),

                "num_turns":
                    len(sub),
            }
        )

        chunk_index += 1
        start += stride

    return docs


def build_raw_c2(turns):

    return make_chunks(
        turns,
        chunk_size=2,
        stride=2,
    )


def build_raw_c6_no_overlap(
    turns,
):

    return make_chunks(
        turns,
        chunk_size=6,
        stride=6,
    )


def build_raw_c6_overlap(
    turns,
    stride,
):

    return make_chunks(
        turns,
        chunk_size=6,
        stride=stride,
    )


# =============================================================================
# SESSION SUMMARIES
# =============================================================================

def build_summary_docs(sample):

    conv = sample[
        "conversation"
    ]

    summaries = sample.get(
        "session_summary",
        {}
    )

    docs = []

    for session_num in (
        get_session_numbers(conv)
    ):

        key = (
            f"session_{session_num}_summary"
        )

        if key not in summaries:
            continue

        summary = summaries[
            key
        ]

        if isinstance(
            summary,
            list,
        ):

            summary = "\n".join(
                str(x)
                for x in summary
            )

        summary = str(
            summary
        ).strip()

        if not summary:
            continue

        session_id = (
            f"session_{session_num}"
        )

        date = conv.get(
            f"session_{session_num}_date_time"
        )

        docs.append(
            {
                "id":
                    key,

                "text":
                    (
                        f"SESSION: {session_id}\n"
                        f"DATE: {date}\n"
                        f"{summary}"
                    ),

                "session_ids": [
                    session_id
                ],

                "dia_ids":
                    [],

                "date":
                    date,
            }
        )

    return docs


def build_memory_docs(
    sample,
    overlap_stride,
):

    turns = (
        flatten_conversation(
            sample
        )
    )

    return {

        RAW_C2:
            build_raw_c2(
                turns
            ),

        RAW_C6_NO:
            build_raw_c6_no_overlap(
                turns
            ),

        RAW_C6_OV:
            build_raw_c6_overlap(
                turns,
                overlap_stride,
            ),

        SUMMARY:
            build_summary_docs(
                sample
            ),
    }


# =============================================================================
# BGE-M3
# =============================================================================

class BGEM3Retriever:

    def __init__(
        self,
        model_name,
        batch_size=32,
        max_length=8192,
    ):

        from FlagEmbedding import (
            BGEM3FlagModel
        )

        print("=" * 100)
        print("LOAD LOCAL BGE-M3")
        print("=" * 100)

        print(
            f"model={model_name}"
        )

        self.model = (
            BGEM3FlagModel(
                model_name,
                use_fp16=True,
            )
        )

        self.batch_size = (
            batch_size
        )

        self.max_length = (
            max_length
        )

    def encode(self, texts):

        output = self.model.encode(
            texts,

            batch_size=
                self.batch_size,

            max_length=
                self.max_length,
        )

        vectors = np.asarray(
            output[
                "dense_vecs"
            ],
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        return (
            vectors
            /
            np.maximum(
                norms,
                1e-12,
            )
        )

    def retrieve(
        self,
        query,
        docs,
        top_k,
    ):

        if not docs:
            return []

        qvec = self.encode(
            [query]
        )[0]

        dvecs = self.encode(
            [
                doc["text"]
                for doc in docs
            ]
        )

        scores = (
            dvecs @ qvec
        )

        order = (
            np.argsort(
                -scores
            )
            [:top_k]
        )

        results = []

        for rank, idx in enumerate(
            order,
            1,
        ):

            item = dict(
                docs[
                    int(idx)
                ]
            )

            item[
                "rank"
            ] = rank

            item[
                "score"
            ] = float(
                scores[idx]
            )

            results.append(
                item
            )

        return results


# =============================================================================
# OPENROUTER
# =============================================================================

class OpenRouterClient:

    def __init__(
        self,
        api_key,
        timeout=60,
        max_retries=3,
    ):

        self.api_key = (
            api_key
        )

        self.timeout = (
            timeout
        )

        self.max_retries = (
            max_retries
        )

        self.local = (
            threading.local()
        )

    def _session(self):

        if not hasattr(
            self.local,
            "session",
        ):

            session = (
                requests.Session()
            )

            session.headers.update(
                {
                    "Authorization":
                        (
                            "Bearer "
                            + self.api_key
                        ),

                    "Content-Type":
                        "application/json",
                }
            )

            self.local.session = (
                session
            )

        return (
            self.local.session
        )

    def chat(
        self,
        model,
        messages,
        max_tokens=128,
        temperature=0.0,
    ):

        payload = {
            "model":
                model,

            "messages":
                messages,

            "temperature":
                temperature,

            "max_tokens":
                max_tokens,

            # We are testing memory strategy,
            # not extended reasoning.
            "reasoning": {
                "enabled": False
            },
        }

        last_error = None

        for attempt in range(
            self.max_retries
        ):

            start = (
                time.time()
            )

            try:

                response = (
                    self._session()
                    .post(
                        OPENROUTER_URL,
                        json=payload,
                        timeout=(
                            15,
                            self.timeout,
                        ),
                    )
                )

                elapsed = (
                    time.time()
                    - start
                )

                if (
                    response.status_code
                    == 429
                ):

                    wait = min(
                        2 ** attempt,
                        10,
                    )

                    print(
                        f"[429] "
                        f"sleep={wait}s",
                        flush=True,
                    )

                    time.sleep(wait)
                    continue

                response.raise_for_status()

                obj = (
                    response.json()
                )

                message = (
                    obj["choices"][0]
                    ["message"]
                )

                content = (
                    message.get(
                        "content",
                        "",
                    )
                    or ""
                )

                content = (
                    str(content)
                    .replace(
                        "</think>",
                        "",
                    )
                    .strip()
                )

                return content

            except Exception as e:

                last_error = e

                print(
                    f"[OpenRouter ERROR] "
                    f"{model} "
                    f"{type(e).__name__}: "
                    f"{e}",
                    flush=True,
                )

                if (
                    attempt
                    < self.max_retries - 1
                ):

                    time.sleep(
                        min(
                            2 ** attempt,
                            5,
                        )
                    )

        raise RuntimeError(
            f"OpenRouter failed: "
            f"{last_error}"
        )


# =============================================================================
# RETRIEVAL FORMAT
# =============================================================================

def format_retrieval(
    title,
    results,
):

    blocks = [
        f"### {title}"
    ]

    for item in results:

        blocks.append(
            (
                f"[RANK {item['rank']} | "
                f"SCORE {item['score']:.4f}]\n"
                f"{item['text']}"
            )
        )

    return "\n\n".join(
        blocks
    )


def format_preview(
    results,
    chars_per_item=900,
):

    blocks = []

    for item in results:

        blocks.append(
            (
                f"RANK={item['rank']}\n"
                f"SCORE={item['score']:.4f}\n"
                f"{item['text'][:chars_per_item]}"
            )
        )

    return "\n\n".join(
        blocks
    )


# =============================================================================
# EM-FRIENDLY QA PROMPT
# =============================================================================

QA_SYSTEM = """
You answer short-answer questions using only the supplied past conversation.

Your output is evaluated with exact-match style metrics.

CRITICAL OUTPUT RULES:

1. Return ONLY the shortest answer span that answers the question.
2. Do NOT write a full sentence unless the answer itself must be a sentence.
3. Do NOT repeat the question.
4. Do NOT write phrases such as:
   - "The answer is"
   - "According to the conversation"
   - "He said"
   - "She said"
   - "It was"
5. Do NOT explain or justify your answer.
6. Do NOT add surrounding context.
7. If the answer is a list, output only the list items in a short natural phrase.
8. Preserve the wording used in the memory whenever possible.
9. Use only supplied memory.
10. If the information is genuinely absent, output exactly:
    unknown

Examples:

Question:
What is John's favorite color?

Good:
blue

Bad:
John's favorite color is blue.


Question:
What did the system help track?

Good:
inventory, resources, and donations

Bad:
The system helped track inventory, resources, and donations.


Question:
What type of seminars is John conducting?

Good:
sports and marketing seminars

Bad:
John is conducting sports and marketing seminars.

Return the minimal answer span only.
""".strip()


def answer_question(
    client,
    model,
    question,
    context,
):

    prompt = f"""
MEMORY:
{context}

QUESTION:
{question}

SHORT ANSWER:
""".strip()

    return client.chat(
        model=model,

        messages=[
            {
                "role":
                    "system",

                "content":
                    QA_SYSTEM,
            },
            {
                "role":
                    "user",

                "content":
                    prompt,
            },
        ],

        max_tokens=80,

        temperature=0.0,
    )


# =============================================================================
# ROUTER 1
# =============================================================================

def choose_best_raw(
    client,
    model,
    question,
    retrieved,
):

    prompt = f"""
You are choosing which RAW memory retrieval representation is most likely
to contain the information required to answer the question.

All candidates contain exactly TOP-5 retrieved items.

Choices:

raw_c2
- 2-turn non-overlapping chunks
- focused evidence
- less surrounding conversational context

raw_c6_no_overlap
- 6-turn non-overlapping chunks
- larger local context
- can lose evidence around chunk boundaries

raw_c6_overlap
- 6-turn chunks with overlap
- larger context
- better boundary coverage
- may contain duplicate evidence

Choose based on actual retrieved evidence, not merely theoretical advantages.

Do NOT answer the question.

Return EXACTLY one JSON object and nothing else:

{{
  "choice": "raw_c2",
  "reason": "brief reason"
}}

Valid choices:
raw_c2
raw_c6_no_overlap
raw_c6_overlap


QUESTION:
{question}


RAW_C2:
{format_preview(retrieved[RAW_C2])}


RAW_C6_NO_OVERLAP:
{format_preview(retrieved[RAW_C6_NO])}


RAW_C6_OVERLAP:
{format_preview(retrieved[RAW_C6_OV])}
""".strip()

    output = client.chat(
        model=model,

        messages=[
            {
                "role":
                    "user",

                "content":
                    prompt,
            }
        ],

        max_tokens=100,

        temperature=0.0,
    )

    match = re.search(
        r'"choice"\s*:\s*"'
        r'(raw_c2|raw_c6_no_overlap|raw_c6_overlap)'
        r'"',
        output,
        flags=re.I,
    )

    if match:

        choice = (
            match.group(1)
            .lower()
        )

    else:

        low = output.lower()

        if (
            RAW_C6_OV
            in low
        ):
            choice = RAW_C6_OV

        elif (
            RAW_C6_NO
            in low
        ):
            choice = RAW_C6_NO

        elif RAW_C2 in low:
            choice = RAW_C2

        else:
            choice = RAW_C6_OV

    reason_match = re.search(
        r'"reason"\s*:\s*"([^"]+)"',
        output,
        flags=re.I,
    )

    reason = (
        reason_match.group(1)
        if reason_match
        else shorten(
            output,
            200,
        )
    )

    return {
        "choice":
            choice,

        "reason":
            reason,

        "raw_output":
            output,
    }


# =============================================================================
# ROUTER 2
# =============================================================================

def choose_raw_vs_summary(
    client,
    model,
    question,
    raw_choice,
    raw_results,
    summary_results,
):

    prompt = f"""
Choose which retrieved memory representation is more likely to support
the exact short answer to the question.

Candidate RAW:
- original dialogue chunks
- preserves exact detail
- may contain distracting text

Candidate SUMMARY:
- existing session-level summaries
- compressed and less noisy
- may omit exact details

Inspect the ACTUAL retrieved content below.

Choose RAW if the raw retrieved text contains clearer or more exact evidence.

Choose SUMMARY if the summary contains the answer more directly or reliably.

Do NOT answer the user's question.

Return EXACTLY:

{{
  "choice": "raw",
  "reason": "brief reason"
}}

or

{{
  "choice": "summary",
  "reason": "brief reason"
}}


QUESTION:
{question}

RAW STRATEGY:
{raw_choice}

RAW TOP-5:
{format_preview(raw_results)}

SUMMARY TOP-5:
{format_preview(summary_results)}
""".strip()

    output = client.chat(
        model=model,

        messages=[
            {
                "role":
                    "user",

                "content":
                    prompt,
            }
        ],

        max_tokens=100,

        temperature=0.0,
    )

    match = re.search(
        r'"choice"\s*:\s*"(raw|summary)"',
        output,
        flags=re.I,
    )

    if match:

        choice = (
            match.group(1)
            .lower()
        )

    else:

        low = output.lower()

        if (
            "summary"
            in low
        ):
            choice = "summary"

        else:
            choice = "raw"

    reason_match = re.search(
        r'"reason"\s*:\s*"([^"]+)"',
        output,
        flags=re.I,
    )

    reason = (
        reason_match.group(1)
        if reason_match
        else shorten(
            output,
            200,
        )
    )

    return {
        "choice":
            choice,

        "reason":
            reason,

        "raw_output":
            output,
    }


# =============================================================================
# SELF ROUTER ALL
# =============================================================================

SELF_ROUTER_SYSTEM = """
You answer short-answer questions using several retrieved representations
of the same conversation memory.

You receive:

- RAW_C2
- RAW_C6_NO_OVERLAP
- RAW_C6_OVERLAP
- SESSION_SUMMARY

First internally determine which evidence is most reliable.
You may combine evidence across representations.

Your final output is evaluated with exact-match style metrics.

CRITICAL FINAL OUTPUT RULES:

- Return ONLY the shortest answer span.
- Do NOT output a full explanatory sentence.
- Do NOT repeat the question.
- Do NOT say "The answer is".
- Do NOT explain which memory representation you used.
- Do NOT mention retrieval.
- Preserve source wording when possible.
- For lists, return only the required list.
- If genuinely absent, output exactly: unknown

Examples:

Question:
What is Jon's favorite style of painting?

Good:
Contemporary

Bad:
Jon's favorite style of painting is Contemporary.


Question:
What did the system help track?

Good:
inventory, resources, and donations

Bad:
The system helped track inventory, resources, and donations.

Return only the minimal answer span.
""".strip()


def self_router_all(
    client,
    model,
    question,
    contexts,
):

    prompt = f"""
QUESTION:
{question}


================ RAW_C2 TOP-5 ================

{contexts[RAW_C2]}


================ RAW_C6_NO_OVERLAP TOP-5 ================

{contexts[RAW_C6_NO]}


================ RAW_C6_OVERLAP TOP-5 ================

{contexts[RAW_C6_OV]}


================ SESSION_SUMMARY TOP-5 ================

{contexts[SUMMARY]}


SHORT ANSWER:
""".strip()

    return client.chat(
        model=model,

        messages=[
            {
                "role":
                    "system",

                "content":
                    SELF_ROUTER_SYSTEM,
            },
            {
                "role":
                    "user",

                "content":
                    prompt,
            },
        ],

        max_tokens=80,

        temperature=0.0,
    )


# =============================================================================
# PARALLEL BASELINES
# =============================================================================

def run_baselines_parallel(
    client,
    model,
    question,
    contexts,
    max_workers,
):

    outputs = {}

    workers = min(
        max_workers,
        len(BASE_METHODS),
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        future_map = {}

        for method in (
            BASE_METHODS
        ):

            future = (
                executor.submit(
                    answer_question,
                    client,
                    model,
                    question,
                    contexts[
                        method
                    ],
                )
            )

            future_map[
                future
            ] = method

        for future in (
            as_completed(
                future_map
            )
        ):

            method = (
                future_map[
                    future
                ]
            )

            try:

                outputs[
                    method
                ] = (
                    future.result()
                )

            except Exception as e:

                print(
                    f"[ERROR] "
                    f"{method}: "
                    f"{e}",
                    flush=True,
                )

                outputs[
                    method
                ] = "__ERROR__"

    return outputs


# =============================================================================
# QA EXAMPLES
# =============================================================================

def build_qa_examples(data):

    examples = []

    for sample in data:

        sample_id = (
            sample.get(
                "sample_id",
                "unknown",
            )
        )

        for qa_index, qa in enumerate(
            sample.get(
                "qa",
                [],
            )
        ):

            gold = qa.get(
                "answer"
            )

            if gold is None:

                gold = (
                    qa.get(
                        "adversarial_answer"
                    )
                )

            if gold is None:
                continue

            examples.append(
                {
                    "sample_id":
                        sample_id,

                    "qa_index":
                        qa_index,

                    "question":
                        qa["question"],

                    "answer":
                        str(gold),

                    "category":
                        qa.get(
                            "category",
                            "unknown",
                        ),

                    "evidence":
                        qa.get(
                            "evidence",
                            [],
                        ),

                    "sample":
                        sample,
                }
            )

    return examples


def stratified_sample(
    examples,
    n,
    seed,
):

    rng = random.Random(
        seed
    )

    groups = defaultdict(
        list
    )

    for x in examples:

        groups[
            str(
                x["category"]
            )
        ].append(x)

    for group in (
        groups.values()
    ):
        rng.shuffle(
            group
        )

    categories = sorted(
        groups.keys()
    )

    base = (
        n // len(categories)
    )

    remainder = (
        n % len(categories)
    )

    selected = []

    for idx, category in enumerate(
        categories
    ):

        count = (
            base
            + (
                1
                if idx < remainder
                else 0
            )
        )

        selected.extend(
            groups[
                category
            ][:count]
        )

    if len(selected) < n:

        used = {
            (
                x["sample_id"],
                x["qa_index"],
            )
            for x in selected
        }

        left = [
            x
            for x in examples
            if (
                x["sample_id"],
                x["qa_index"],
            )
            not in used
        ]

        rng.shuffle(
            left
        )

        selected.extend(
            left[
                :n-len(selected)
            ]
        )

    rng.shuffle(
        selected
    )

    return selected[:n]


# =============================================================================
# RETRIEVAL HIT
# =============================================================================

def raw_evidence_hit(
    retrieved,
    evidence,
):

    gold = set(
        evidence or []
    )

    if not gold:
        return None

    found = set()

    for item in retrieved:

        found.update(
            item.get(
                "dia_ids",
                [],
            )
        )

    return int(
        bool(
            found & gold
        )
    )


def summary_evidence_hit(
    retrieved,
    evidence,
):

    gold_sessions = set()

    for evidence_id in (
        evidence or []
    ):

        m = re.match(
            r"D(\d+):",
            str(evidence_id),
        )

        if m:

            gold_sessions.add(
                f"session_"
                f"{int(m.group(1))}"
            )

    if not gold_sessions:
        return None

    found_sessions = set()

    for item in retrieved:

        found_sessions.update(
            item.get(
                "session_ids",
                [],
            )
        )

    return int(
        bool(
            found_sessions
            & gold_sessions
        )
    )


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate(rows):

    stats = {}

    for method in METHODS:

        vals = [
            row["results"][method]
            for row in rows
        ]

        stats[
            method
        ] = {

            "n":
                len(vals),

            "em":
                safe_mean(
                    [
                        x["em"]
                        for x in vals
                    ]
                ),

            "strict_em":
                safe_mean(
                    [
                        x["strict_em"]
                        for x in vals
                    ]
                ),

            "f1":
                safe_mean(
                    [
                        x["f1"]
                        for x in vals
                    ]
                ),

            "avg_input_chars":
                safe_mean(
                    [
                        x[
                            "input_chars"
                        ]
                        for x in vals
                    ]
                ),
        }

    return stats


def retrieval_stats(rows):

    output = {}

    for method in (
        BASE_METHODS
    ):

        hits = [
            row[
                "retrieval"
            ][method][
                "evidence_hit"
            ]
            for row in rows
            if (
                row[
                    "retrieval"
                ][method][
                    "evidence_hit"
                ]
                is not None
            )
        ]

        output[
            method
        ] = (
            safe_mean(hits)
            if hits
            else None
        )

    return output


# =============================================================================
# ORACLE
# =============================================================================

def compute_oracles(rows):

    raw_em = []
    raw_strict = []
    raw_f1 = []

    all_em = []
    all_strict = []
    all_f1 = []

    for row in rows:

        raw_em.append(
            max(
                row["results"][m]["em"]
                for m in RAW_METHODS
            )
        )

        raw_strict.append(
            max(
                row["results"][m]["strict_em"]
                for m in RAW_METHODS
            )
        )

        raw_f1.append(
            max(
                row["results"][m]["f1"]
                for m in RAW_METHODS
            )
        )

        all_em.append(
            max(
                row["results"][m]["em"]
                for m in BASE_METHODS
            )
        )

        all_strict.append(
            max(
                row["results"][m]["strict_em"]
                for m in BASE_METHODS
            )
        )

        all_f1.append(
            max(
                row["results"][m]["f1"]
                for m in BASE_METHODS
            )
        )

    return {

        "oracle_raw": {
            "em":
                safe_mean(
                    raw_em
                ),

            "strict_em":
                safe_mean(
                    raw_strict
                ),

            "f1":
                safe_mean(
                    raw_f1
                ),
        },

        "oracle_all": {
            "em":
                safe_mean(
                    all_em
                ),

            "strict_em":
                safe_mean(
                    all_strict
                ),

            "f1":
                safe_mean(
                    all_f1
                ),
        },
    }


# =============================================================================
# PRINT
# =============================================================================

def print_cumulative(
    rows,
    current,
    total,
):

    stats = aggregate(
        rows
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        f"CUMULATIVE RESULT "
        f"[{current}/{total}]"
    )

    print(
        "=" * 120
    )

    print(
        f"{'METHOD':30s}"
        f"{'N':>6s}"
        f"{'EM':>10s}"
        f"{'STRICT':>10s}"
        f"{'F1':>10s}"
        f"{'AVG_CHARS':>14s}"
    )

    print(
        "-" * 120
    )

    for method in METHODS:

        s = stats[
            method
        ]

        print(
            f"{method:30s}"
            f"{s['n']:6d}"
            f"{s['em']:10.4f}"
            f"{s['strict_em']:10.4f}"
            f"{s['f1']:10.4f}"
            f"{s['avg_input_chars']:14.0f}"
        )

    print(
        "-" * 120
    )

    # ----------------------------------------------------------------
    # Retrieval
    # ----------------------------------------------------------------

    rstats = (
        retrieval_stats(
            rows
        )
    )

    print(
        "\n[CUM RETRIEVAL EVIDENCE HIT@5]"
    )

    for method in (
        BASE_METHODS
    ):

        value = rstats[
            method
        ]

        text = (
            f"{value:.4f}"
            if value is not None
            else "N/A"
        )

        print(
            f"  {method:30s}"
            f"{text}"
        )

    # ----------------------------------------------------------------
    # Router
    # ----------------------------------------------------------------

    raw_dist = Counter(
        row[
            "router"
        ][
            "raw_choice"
        ]
        for row in rows
    )

    rep_dist = Counter(
        row[
            "router"
        ][
            "representation_choice"
        ]
        for row in rows
    )

    print(
        "\n[CUM RAW ROUTER]"
    )

    for method in (
        RAW_METHODS
    ):

        count = (
            raw_dist.get(
                method,
                0,
            )
        )

        print(
            f"  {method:30s}"
            f"{count:4d} "
            f"({count/len(rows):.1%})"
        )

    print(
        "\n[CUM RAW VS SUMMARY ROUTER]"
    )

    for key in [
        "raw",
        "summary",
    ]:

        count = (
            rep_dist.get(
                key,
                0,
            )
        )

        print(
            f"  {key:30s}"
            f"{count:4d} "
            f"({count/len(rows):.1%})"
        )

    # ----------------------------------------------------------------
    # Oracle
    # ----------------------------------------------------------------

    oracle = (
        compute_oracles(
            rows
        )
    )

    print(
        "\n[CUM ORACLE]"
    )

    print(
        "  oracle_raw "
        f"EM={oracle['oracle_raw']['em']:.4f} "
        f"STRICT={oracle['oracle_raw']['strict_em']:.4f} "
        f"F1={oracle['oracle_raw']['f1']:.4f}"
    )

    print(
        "  oracle_all "
        f"EM={oracle['oracle_all']['em']:.4f} "
        f"STRICT={oracle['oracle_all']['strict_em']:.4f} "
        f"F1={oracle['oracle_all']['f1']:.4f}"
    )

    best_em = max(
        METHODS,
        key=lambda m:
            stats[m]["em"],
    )

    best_f1 = max(
        METHODS,
        key=lambda m:
            stats[m]["f1"],
    )

    print(
        "\n[CURRENT BEST]"
    )

    print(
        f"  EM : "
        f"{best_em} "
        f"{stats[best_em]['em']:.4f}"
    )

    print(
        f"  F1 : "
        f"{best_f1} "
        f"{stats[best_f1]['f1']:.4f}"
    )


# =============================================================================
# SAVE
# =============================================================================

def save_details(
    rows,
    output_dir,
):

    path = (
        Path(output_dir)
        / "details.jsonl"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        for row in rows:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_summary_csv(
    rows,
    output_dir,
):

    stats = (
        aggregate(
            rows
        )
    )

    path = (
        Path(output_dir)
        / "summary.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = (
            csv.writer(f)
        )

        writer.writerow(
            [
                "method",
                "n",
                "em",
                "strict_em",
                "f1",
                "avg_input_chars",
            ]
        )

        for method in METHODS:

            s = stats[
                method
            ]

            writer.writerow(
                [
                    method,
                    s["n"],
                    s["em"],
                    s["strict_em"],
                    s["f1"],
                    s[
                        "avg_input_chars"
                    ],
                ]
            )


def save_by_category(
    rows,
    output_dir,
):

    path = (
        Path(output_dir)
        / "by_category.csv"
    )

    categories = sorted(
        {
            str(
                row[
                    "category"
                ]
            )
            for row in rows
        }
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = (
            csv.writer(f)
        )

        writer.writerow(
            [
                "category",
                "method",
                "n",
                "em",
                "strict_em",
                "f1",
            ]
        )

        for category in (
            categories
        ):

            subset = [
                row
                for row in rows
                if (
                    str(
                        row[
                            "category"
                        ]
                    )
                    == category
                )
            ]

            stats = (
                aggregate(
                    subset
                )
            )

            for method in METHODS:

                s = stats[
                    method
                ]

                writer.writerow(
                    [
                        category,
                        method,
                        s["n"],
                        s["em"],
                        s[
                            "strict_em"
                        ],
                        s["f1"],
                    ]
                )


def save_router_choices(
    rows,
    output_dir,
):

    path = (
        Path(output_dir)
        / "router_choices.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = (
            csv.writer(f)
        )

        writer.writerow(
            [
                "sample_id",
                "qa_index",
                "category",
                "raw_choice",
                "raw_reason",
                "representation_choice",
                "representation_reason",
                "final_source",
            ]
        )

        for row in rows:

            r = row[
                "router"
            ]

            writer.writerow(
                [
                    row[
                        "sample_id"
                    ],

                    row[
                        "qa_index"
                    ],

                    row[
                        "category"
                    ],

                    r[
                        "raw_choice"
                    ],

                    r[
                        "raw_reason"
                    ],

                    r[
                        "representation_choice"
                    ],

                    r[
                        "representation_reason"
                    ],

                    r[
                        "final_source"
                    ],
                ]
            )


def save_paired(
    rows,
    output_dir,
):

    path = (
        Path(output_dir)
        / "paired.csv"
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = (
            csv.writer(f)
        )

        writer.writerow(
            [
                "method_a",
                "method_b",
                "a_wins_em",
                "b_wins_em",
                "ties_em",
                "a_wins_f1",
                "b_wins_f1",
                "ties_f1",
                "mean_f1_delta",
            ]
        )

        for i, a in enumerate(
            METHODS
        ):

            for b in (
                METHODS[
                    i + 1:
                ]
            ):

                a_em = 0
                b_em = 0
                tie_em = 0

                a_f1 = 0
                b_f1 = 0
                tie_f1 = 0

                deltas = []

                for row in rows:

                    ra = (
                        row[
                            "results"
                        ][a]
                    )

                    rb = (
                        row[
                            "results"
                        ][b]
                    )

                    if (
                        ra["em"]
                        > rb["em"]
                    ):
                        a_em += 1

                    elif (
                        rb["em"]
                        > ra["em"]
                    ):
                        b_em += 1

                    else:
                        tie_em += 1

                    if (
                        ra["f1"]
                        > rb["f1"]
                        + 1e-9
                    ):
                        a_f1 += 1

                    elif (
                        rb["f1"]
                        > ra["f1"]
                        + 1e-9
                    ):
                        b_f1 += 1

                    else:
                        tie_f1 += 1

                    deltas.append(
                        ra["f1"]
                        - rb["f1"]
                    )

                writer.writerow(
                    [
                        a,
                        b,
                        a_em,
                        b_em,
                        tie_em,
                        a_f1,
                        b_f1,
                        tie_f1,
                        safe_mean(
                            deltas
                        ),
                    ]
                )


def save_all(
    rows,
    output_dir,
):

    save_details(
        rows,
        output_dir,
    )

    save_summary_csv(
        rows,
        output_dir,
    )

    save_by_category(
        rows,
        output_dir,
    )

    save_router_choices(
        rows,
        output_dir,
    )

    save_paired(
        rows,
        output_dir,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--dataset",
        default="./locomo10.json",
    )

    parser.add_argument(
        "--output-dir",
        default="./results_locomo_final",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--c6-overlap-stride",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--bge-model",
        default=DEFAULT_BGE_MODEL,
    )

    parser.add_argument(
        "--bge-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--qa-model",
        default=DEFAULT_QA_MODEL,
    )

    parser.add_argument(
        "--router-model",
        default=DEFAULT_ROUTER_MODEL,
    )

    parser.add_argument(
        "--self-router-model",
        default=DEFAULT_SELF_ROUTER_MODEL,
    )

    args = parser.parse_args()

    if args.max_workers < 1:

        raise ValueError(
            "--max-workers >= 1"
        )

    api_key = (
        os.environ.get(
            "OPENROUTER_API_KEY"
        )
    )

    if not api_key:

        raise RuntimeError(
            "export OPENROUTER_API_KEY='...'"
        )

    ensure_dir(
        args.output_dir
    )

    # =========================================================================
    # CONFIG
    # =========================================================================

    print(
        "=" * 120
    )

    print(
        "CONFIG"
    )

    print(
        "=" * 120
    )

    print(
        f"num_examples        : "
        f"{args.num_examples}"
    )

    print(
        f"top_k               : "
        f"{args.top_k}"
    )

    print(
        f"max_workers         : "
        f"{args.max_workers}"
    )

    print(
        f"C2                  : "
        f"size=2 stride=2"
    )

    print(
        f"C6 NO               : "
        f"size=6 stride=6"
    )

    print(
        f"C6 OVERLAP          : "
        f"size=6 stride="
        f"{args.c6_overlap_stride}"
    )

    print(
        f"BGE                 : "
        f"{args.bge_model}"
    )

    print(
        f"QA                  : "
        f"{args.qa_model}"
    )

    print(
        f"Router              : "
        f"{args.router_model}"
    )

    print(
        f"Self Router         : "
        f"{args.self_router_model}"
    )

    # =========================================================================
    # DATA
    # =========================================================================

    download_locomo(
        args.dataset
    )

    data = (
        load_locomo(
            args.dataset
        )
    )

    all_examples = (
        build_qa_examples(
            data
        )
    )

    print(
        f"[QA] total="
        f"{len(all_examples)}"
    )

    examples = (
        stratified_sample(
            all_examples,
            args.num_examples,
            args.seed,
        )
    )

    print(
        "\n[EVAL SAMPLE]"
    )

    print(
        "categories:",
        dict(
            Counter(
                str(
                    x["category"]
                )
                for x in examples
            )
        ),
    )

    # =========================================================================
    # MODELS
    # =========================================================================

    retriever = (
        BGEM3Retriever(
            args.bge_model,
            batch_size=(
                args.bge_batch_size
            ),
        )
    )

    client = (
        OpenRouterClient(
            api_key
        )
    )

    doc_cache = {}

    rows = []

    # =========================================================================
    # EVAL LOOP
    # =========================================================================

    for idx, ex in enumerate(
        examples,
        1,
    ):

        start_time = (
            time.time()
        )

        sample = ex[
            "sample"
        ]

        sample_id = ex[
            "sample_id"
        ]

        qa_index = ex[
            "qa_index"
        ]

        category = ex[
            "category"
        ]

        question = ex[
            "question"
        ]

        gold = ex[
            "answer"
        ]

        evidence = ex[
            "evidence"
        ]

        print(
            "\n\n"
            + "=" * 120
        )

        print(
            f"[{idx:03d}/"
            f"{len(examples):03d}] "
            f"sample={sample_id} "
            f"qa={qa_index} "
            f"category={category}"
        )

        print(
            "=" * 120
        )

        print(
            f"Q    : {question}"
        )

        print(
            f"GOLD : {gold}"
        )

        # ---------------------------------------------------------------------
        # docs
        # ---------------------------------------------------------------------

        if (
            sample_id
            not in doc_cache
        ):

            doc_cache[
                sample_id
            ] = (
                build_memory_docs(
                    sample,
                    args.c6_overlap_stride,
                )
            )

            print(
                "\n[INDEX SIZE]"
            )

            for method in (
                BASE_METHODS
            ):

                print(
                    f"  {method:30s}"
                    f"{len(doc_cache[sample_id][method]):6d}"
                )

        docs = (
            doc_cache[
                sample_id
            ]
        )

        # ---------------------------------------------------------------------
        # retrieval
        # ---------------------------------------------------------------------

        retrieved = {}
        retrieval_hits = {}

        print(
            "\n[RETRIEVAL]"
        )

        for method in (
            BASE_METHODS
        ):

            t0 = time.time()

            ret = (
                retriever.retrieve(
                    question,
                    docs[
                        method
                    ],
                    top_k=args.top_k,
                )
            )

            retrieved[
                method
            ] = ret

            if (
                method == SUMMARY
            ):

                hit = (
                    summary_evidence_hit(
                        ret,
                        evidence,
                    )
                )

            else:

                hit = (
                    raw_evidence_hit(
                        ret,
                        evidence,
                    )
                )

            retrieval_hits[
                method
            ] = hit

            print(
                f"  {method:30s}"
                f"hit@{args.top_k}="
                f"{str(hit):5s}"
                f"time="
                f"{time.time()-t0:.2f}s"
            )

            for item in ret:

                print(
                    f"      "
                    f"#{item['rank']} "
                    f"{item['score']:.4f} "
                    f"{shorten(item['text'], 115)}"
                )

        # ---------------------------------------------------------------------
        # contexts
        # ---------------------------------------------------------------------

        contexts = {

            RAW_C2:
                format_retrieval(
                    "RAW C2 TOP-5",
                    retrieved[
                        RAW_C2
                    ],
                ),

            RAW_C6_NO:
                format_retrieval(
                    "RAW C6 NO OVERLAP TOP-5",
                    retrieved[
                        RAW_C6_NO
                    ],
                ),

            RAW_C6_OV:
                format_retrieval(
                    "RAW C6 OVERLAP TOP-5",
                    retrieved[
                        RAW_C6_OV
                    ],
                ),

            SUMMARY:
                format_retrieval(
                    "SESSION SUMMARY TOP-5",
                    retrieved[
                        SUMMARY
                    ],
                ),
        }

        # ---------------------------------------------------------------------
        # BASELINES
        # ---------------------------------------------------------------------

        print(
            "\n[BASELINE QA]"
        )

        qa_start = (
            time.time()
        )

        predictions = (
            run_baselines_parallel(
                client,
                args.qa_model,
                question,
                contexts,
                args.max_workers,
            )
        )

        print(
            f"  parallel time="
            f"{time.time()-qa_start:.2f}s"
        )

        for method in (
            BASE_METHODS
        ):

            pred = (
                predictions[
                    method
                ]
            )

            em = (
                relaxed_exact_match(
                    pred,
                    gold,
                )
            )

            strict = (
                strict_exact_match(
                    pred,
                    gold,
                )
            )

            f1 = (
                token_f1(
                    pred,
                    gold,
                )
            )

            print(
                f"  {method:30s}"
                f"EM={em:.0f} "
                f"STRICT={strict:.0f} "
                f"F1={f1:.4f} "
                f"PRED={shorten(pred, 100)}"
            )

        # ---------------------------------------------------------------------
        # ROUTER 1
        # ---------------------------------------------------------------------

        print(
            "\n[ROUTER STAGE 1: RAW]"
        )

        raw_route = (
            choose_best_raw(
                client,
                args.router_model,
                question,
                retrieved,
            )
        )

        raw_choice = (
            raw_route[
                "choice"
            ]
        )

        print(
            f"  choice="
            f"{raw_choice}"
        )

        print(
            f"  reason="
            f"{raw_route['reason']}"
        )

        # ---------------------------------------------------------------------
        # ROUTER 2
        # ---------------------------------------------------------------------

        print(
            "\n[ROUTER STAGE 2: RAW VS SUMMARY]"
        )

        rep_route = (
            choose_raw_vs_summary(
                client,
                args.router_model,
                question,
                raw_choice,
                retrieved[
                    raw_choice
                ],
                retrieved[
                    SUMMARY
                ],
            )
        )

        rep_choice = (
            rep_route[
                "choice"
            ]
        )

        print(
            f"  choice="
            f"{rep_choice}"
        )

        print(
            f"  reason="
            f"{rep_route['reason']}"
        )

        if (
            rep_choice
            == "summary"
        ):

            router_source = (
                SUMMARY
            )

        else:

            router_source = (
                raw_choice
            )

        router_context = (
            contexts[
                router_source
            ]
        )

        # ---------------------------------------------------------------------
        # Router QA + Self all in parallel
        # ---------------------------------------------------------------------

        print(
            "\n[ROUTER FINAL + SELF ROUTER ALL]"
        )

        with ThreadPoolExecutor(
            max_workers=min(
                args.max_workers,
                2,
            )
        ) as executor:

            router_future = (
                executor.submit(
                    answer_question,
                    client,
                    args.qa_model,
                    question,
                    router_context,
                )
            )

            self_future = (
                executor.submit(
                    self_router_all,
                    client,
                    args.self_router_model,
                    question,
                    contexts,
                )
            )

            router_pred = (
                router_future.result()
            )

            self_pred = (
                self_future.result()
            )

        predictions[
            ROUTER
        ] = router_pred

        predictions[
            SELF_ALL
        ] = self_pred

        for method, pred in [
            (
                ROUTER,
                router_pred,
            ),
            (
                SELF_ALL,
                self_pred,
            ),
        ]:

            print(
                f"  {method:30s}"
                f"EM="
                f"{relaxed_exact_match(pred,gold):.0f} "
                f"STRICT="
                f"{strict_exact_match(pred,gold):.0f} "
                f"F1="
                f"{token_f1(pred,gold):.4f} "
                f"PRED="
                f"{shorten(pred,100)}"
            )

        # ---------------------------------------------------------------------
        # results
        # ---------------------------------------------------------------------

        results = {}

        for method in METHODS:

            pred = (
                predictions[
                    method
                ]
            )

            if (
                method
                in BASE_METHODS
            ):

                input_chars = (
                    len(
                        contexts[
                            method
                        ]
                    )
                )

            elif (
                method
                == ROUTER
            ):

                input_chars = (
                    len(
                        router_context
                    )
                )

            else:

                input_chars = (
                    sum(
                        len(
                            contexts[m]
                        )
                        for m
                        in BASE_METHODS
                    )
                )

            results[
                method
            ] = {

                "prediction":
                    pred,

                "em":
                    relaxed_exact_match(
                        pred,
                        gold,
                    ),

                "strict_em":
                    strict_exact_match(
                        pred,
                        gold,
                    ),

                "f1":
                    token_f1(
                        pred,
                        gold,
                    ),

                "input_chars":
                    input_chars,
            }

        # ---------------------------------------------------------------------
        # retrieval info
        # ---------------------------------------------------------------------

        retrieval_info = {}

        for method in (
            BASE_METHODS
        ):

            retrieval_info[
                method
            ] = {

                "evidence_hit":
                    retrieval_hits[
                        method
                    ],

                "results": [
                    {
                        "rank":
                            x["rank"],

                        "score":
                            x["score"],

                        "id":
                            x["id"],

                        "session_ids":
                            x.get(
                                "session_ids",
                                [],
                            ),

                        "dia_ids":
                            x.get(
                                "dia_ids",
                                [],
                            ),
                    }

                    for x in (
                        retrieved[
                            method
                        ]
                    )
                ],
            }

        row = {

            "sample_id":
                sample_id,

            "qa_index":
                qa_index,

            "category":
                category,

            "question":
                question,

            "gold":
                gold,

            "evidence":
                evidence,

            "retrieval":
                retrieval_info,

            "router": {

                "raw_choice":
                    raw_choice,

                "raw_reason":
                    raw_route[
                        "reason"
                    ],

                "representation_choice":
                    rep_choice,

                "representation_reason":
                    rep_route[
                        "reason"
                    ],

                "final_source":
                    router_source,
            },

            "results":
                results,

            "elapsed_sec":
                (
                    time.time()
                    - start_time
                ),
        }

        rows.append(
            row
        )

        # ---------------------------------------------------------------------
        # CURRENT
        # ---------------------------------------------------------------------

        print(
            "\n[CURRENT QUESTION]"
        )

        for method in METHODS:

            r = (
                results[
                    method
                ]
            )

            mark = (
                "✓"
                if r["em"] == 1
                else "✗"
            )

            print(
                f"  {method:30s}"
                f"{mark} "
                f"EM={r['em']:.0f} "
                f"STRICT={r['strict_em']:.0f} "
                f"F1={r['f1']:.4f}"
            )

        print(
            f"\n  elapsed="
            f"{row['elapsed_sec']:.2f}s"
        )

        # ---------------------------------------------------------------------
        # save + cumulative
        # ---------------------------------------------------------------------

        save_all(
            rows,
            args.output_dir,
        )

        print_cumulative(
            rows,
            idx,
            len(examples),
        )

    # =========================================================================
    # FINAL
    # =========================================================================

    print(
        "\n\n"
        + "#" * 120
    )

    print(
        "FINAL RESULT"
    )

    print(
        "#" * 120
    )

    print_cumulative(
        rows,
        len(rows),
        len(rows),
    )

    print(
        f"\nSaved under: "
        f"{args.output_dir}"
    )

    print(
        "  details.jsonl"
    )

    print(
        "  summary.csv"
    )

    print(
        "  by_category.csv"
    )

    print(
        "  router_choices.csv"
    )

    print(
        "  paired.csv"
    )


if __name__ == "__main__":
    main()
