#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoCoMo final ablation:
C6 overlap vs Session Summary WITHOUT self-router.

LOCAL
=====
BAAI/bge-m3

OPENROUTER
==========
Final QA only

METHODS
=======
1. c6_overlap_no_selfrouter
   C6 overlap
   -> BGE-M3 TOP-5
   -> NO relevance filtering
   -> final QA

2. summary_no_selfrouter
   Session summaries
   -> BGE-M3 TOP-5
   -> NO relevance filtering
   -> final QA

METRICS
=======
- relaxed EM
- strict EM
- token F1
- retrieval evidence hit@5
- avg input chars

PROMPT
======
Uses the same EM-friendly short-answer QA prompt
from the previous experiments.

FAIR COMPARISON
===============
Recommended:
--eval-set ./results_locomo_final/selected_eval_set.json

This guarantees the exact same 50 QA examples.
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
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"


C6_METHOD = "c6_overlap_no_selfrouter"
SUMMARY_METHOD = "summary_no_selfrouter"

METHODS = [
    C6_METHOD,
    SUMMARY_METHOD,
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
    text = str(text).lower()

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

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
    return float(
        normalize_answer(pred)
        ==
        normalize_answer(gold)
    )


def relaxed_exact_match(pred, gold):
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
    p = normalize_answer(
        pred
    ).split()

    g = normalize_answer(
        gold
    ).split()

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

        match = re.fullmatch(
            r"session_(\d+)",
            key,
        )

        if match:
            result.append(
                int(
                    match.group(1)
                )
            )

    return sorted(
        result
    )


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
        get_session_numbers(
            conv
        )
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
                        parsed["dia_id"],

                    "text":
                        parsed["text"],
                }
            )

    return turns


# =============================================================================
# C6 OVERLAP
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
                    str(
                        turn["date"]
                    )
                )

            metadata.append(
                turn[
                    "session_id"
                ]
            )

            prefix = (
                "["
                + " | ".join(
                    metadata
                )
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
                            t[
                                "session_id"
                            ]
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


def build_c6_overlap(
    turns,
    stride=3,
):
    return make_chunks(
        turns,
        chunk_size=6,
        stride=stride,
    )


# =============================================================================
# SESSION SUMMARIES
# =============================================================================

def build_summary_docs(
    sample,
):
    conv = sample[
        "conversation"
    ]

    summaries = sample.get(
        "session_summary",
        {}
    )

    docs = []

    for session_num in (
        get_session_numbers(
            conv
        )
    ):

        key = (
            f"session_{session_num}_summary"
        )

        if key not in summaries:
            continue

        summary = (
            summaries[
                key
            ]
        )

        if isinstance(
            summary,
            list,
        ):
            summary = (
                "\n".join(
                    str(x)
                    for x in summary
                )
            )

        summary = (
            str(summary)
            .strip()
        )

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
                        f"SUMMARY:\n"
                        f"{summary}"
                    ),

                "session_ids": [
                    session_id
                ],

                "dia_ids":
                    [],
            }
        )

    return docs


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

    def encode(
        self,
        texts,
    ):

        output = (
            self.model.encode(
                texts,
                batch_size=(
                    self.batch_size
                ),
                max_length=(
                    self.max_length
                ),
            )
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
        top_k=5,
    ):

        if not docs:
            return []

        qvec = (
            self.encode(
                [query]
            )[0]
        )

        dvecs = (
            self.encode(
                [
                    doc["text"]
                    for doc in docs
                ]
            )
        )

        scores = (
            dvecs
            @ qvec
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
                scores[
                    idx
                ]
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

    def _session(
        self,
    ):

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
        max_tokens=80,
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

            "reasoning": {
                "enabled":
                    False
            },
        }

        last_error = None

        for attempt in range(
            self.max_retries
        ):

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

                    time.sleep(
                        wait
                    )

                    continue

                response.raise_for_status()

                obj = (
                    response.json()
                )

                content = (
                    obj[
                        "choices"
                    ][0][
                        "message"
                    ].get(
                        "content",
                        "",
                    )
                    or ""
                )

                return (
                    str(content)
                    .replace(
                        "</think>",
                        "",
                    )
                    .strip()
                )

            except Exception as e:

                last_error = e

                print(
                    f"[OpenRouter ERROR] "
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
            str(
                last_error
            )
        )


# =============================================================================
# SAME EM-FRIENDLY QA PROMPT
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


def format_docs(
    docs,
):

    if not docs:
        return (
            "NO RETRIEVED MEMORY"
        )

    blocks = []

    for idx, doc in enumerate(
        docs,
        1,
    ):

        blocks.append(
            (
                f"[MEMORY {idx} | "
                f"RANK {doc.get('rank')} | "
                f"SCORE {doc.get('score', 0):.4f}]\n"
                f"{doc['text']}"
            )
        )

    return "\n\n".join(
        blocks
    )


def answer_question(
    client,
    model,
    question,
    docs,
):

    context = (
        format_docs(
            docs
        )
    )

    prompt = f"""
MEMORY:
{context}

QUESTION:
{question}

SHORT ANSWER:
""".strip()

    prediction = (
        client.chat(
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
    )

    return (
        prediction,
        len(context),
    )


# =============================================================================
# RETRIEVAL HIT
# =============================================================================

def raw_evidence_hit(
    docs,
    evidence,
):

    gold = set(
        evidence or []
    )

    if not gold:
        return None

    retrieved_ids = set()

    for doc in docs:

        retrieved_ids.update(
            doc.get(
                "dia_ids",
                [],
            )
        )

    return int(
        bool(
            retrieved_ids
            & gold
        )
    )


def summary_evidence_hit(
    docs,
    evidence,
):

    gold_sessions = set()

    for evidence_id in (
        evidence or []
    ):

        match = re.match(
            r"D(\d+):",
            str(
                evidence_id
            ),
        )

        if match:

            gold_sessions.add(
                f"session_"
                f"{int(match.group(1))}"
            )

    if not gold_sessions:
        return None

    retrieved_sessions = set()

    for doc in docs:

        retrieved_sessions.update(
            doc.get(
                "session_ids",
                [],
            )
        )

    return int(
        bool(
            retrieved_sessions
            & gold_sessions
        )
    )


# =============================================================================
# QA DATA
# =============================================================================

def build_qa_examples(
    data,
):

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

            gold = (
                qa.get(
                    "answer"
                )
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
                        qa[
                            "question"
                        ],

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

    rng = (
        random.Random(
            seed
        )
    )

    groups = defaultdict(
        list
    )

    for example in examples:

        groups[
            str(
                example[
                    "category"
                ]
            )
        ].append(
            example
        )

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
        n
        // len(categories)
    )

    remainder = (
        n
        % len(categories)
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
                x[
                    "sample_id"
                ],
                x[
                    "qa_index"
                ],
            )
            for x in selected
        }

        leftover = [
            x
            for x in examples
            if (
                x[
                    "sample_id"
                ],
                x[
                    "qa_index"
                ],
            )
            not in used
        ]

        rng.shuffle(
            leftover
        )

        selected.extend(
            leftover[
                :
                n - len(selected)
            ]
        )

    rng.shuffle(
        selected
    )

    return (
        selected[:n]
    )


def load_selected_eval_set(
    path,
    all_examples,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        selected = (
            json.load(
                f
            )
        )

    lookup = {
        (
            x["sample_id"],
            x["qa_index"],
        ):
            x

        for x
        in all_examples
    }

    output = []

    for item in selected:

        key = (
            item[
                "sample_id"
            ],

            item[
                "qa_index"
            ],
        )

        if key not in lookup:

            raise KeyError(
                f"Example not found: "
                f"{key}"
            )

        output.append(
            lookup[
                key
            ]
        )

    return output


# =============================================================================
# AGGREGATE
# =============================================================================

def aggregate(
    rows,
):

    stats = {}

    for method in (
        METHODS
    ):

        vals = [
            row[
                "results"
            ][
                method
            ]
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
                        for x
                        in vals
                    ]
                ),

            "strict_em":
                safe_mean(
                    [
                        x[
                            "strict_em"
                        ]
                        for x
                        in vals
                    ]
                ),

            "f1":
                safe_mean(
                    [
                        x[
                            "f1"
                        ]
                        for x
                        in vals
                    ]
                ),

            "avg_input_chars":
                safe_mean(
                    [
                        x[
                            "input_chars"
                        ]
                        for x
                        in vals
                    ]
                ),
        }

    return stats


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
        + "=" * 112
    )

    print(
        f"CUMULATIVE RESULT "
        f"[{current}/{total}]"
    )

    print(
        "=" * 112
    )

    print(
        f"{'METHOD':34s}"
        f"{'N':>6s}"
        f"{'EM':>10s}"
        f"{'STRICT':>10s}"
        f"{'F1':>10s}"
        f"{'AVG_CHARS':>14s}"
    )

    print(
        "-" * 112
    )

    for method in METHODS:

        s = (
            stats[
                method
            ]
        )

        print(
            f"{method:34s}"
            f"{s['n']:6d}"
            f"{s['em']:10.4f}"
            f"{s['strict_em']:10.4f}"
            f"{s['f1']:10.4f}"
            f"{s['avg_input_chars']:14.0f}"
        )

    print(
        "-" * 112
    )

    # ---------------------------------------------------------
    # Retrieval hit
    # ---------------------------------------------------------

    c6_hits = [
        row[
            "retrieval"
        ][
            "c6_hit"
        ]
        for row in rows
        if (
            row[
                "retrieval"
            ][
                "c6_hit"
            ]
            is not None
        )
    ]

    summary_hits = [
        row[
            "retrieval"
        ][
            "summary_hit"
        ]
        for row in rows
        if (
            row[
                "retrieval"
            ][
                "summary_hit"
            ]
            is not None
        )
    ]

    print(
        "\n[RETRIEVAL HIT@5]"
    )

    print(
        f"  c6_overlap      : "
        f"{safe_mean(c6_hits):.4f}"
    )

    print(
        f"  session_summary : "
        f"{safe_mean(summary_hits):.4f}"
    )

    # ---------------------------------------------------------
    # Pair
    # ---------------------------------------------------------

    c6_only = 0
    summary_only = 0
    both = 0
    both_wrong = 0

    for row in rows:

        a = (
            row[
                "results"
            ][
                C6_METHOD
            ][
                "em"
            ]
        )

        b = (
            row[
                "results"
            ][
                SUMMARY_METHOD
            ][
                "em"
            ]
        )

        if (
            a == 1
            and b == 1
        ):

            both += 1

        elif (
            a == 1
        ):

            c6_only += 1

        elif (
            b == 1
        ):

            summary_only += 1

        else:

            both_wrong += 1

    print(
        "\n[PAIRED EM]"
    )

    print(
        f"  c6_only     = "
        f"{c6_only}"
    )

    print(
        f"  summary_only= "
        f"{summary_only}"
    )

    print(
        f"  both        = "
        f"{both}"
    )

    print(
        f"  both_wrong  = "
        f"{both_wrong}"
    )


# =============================================================================
# SAVE
# =============================================================================

def save_all(
    rows,
    output_dir,
):

    detail_path = (
        Path(
            output_dir
        )
        / "details.jsonl"
    )

    with open(
        detail_path,
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

    stats = aggregate(
        rows
    )

    summary_path = (
        Path(
            output_dir
        )
        / "summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = (
            csv.writer(
                f
            )
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

            s = (
                stats[
                    method
                ]
            )

            writer.writerow(
                [
                    method,
                    s["n"],
                    s["em"],
                    s[
                        "strict_em"
                    ],
                    s["f1"],
                    s[
                        "avg_input_chars"
                    ],
                ]
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
        default=(
            "./results_locomo_no_selfrouter"
        ),
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
        "--eval-set",
        default=None,
        help=(
            "Path to previous "
            "selected_eval_set.json"
        ),
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
        default=(
            DEFAULT_BGE_MODEL
        ),
    )

    parser.add_argument(
        "--bge-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--model",
        default=(
            DEFAULT_MODEL
        ),
    )

    args = (
        parser.parse_args()
    )

    if (
        args.max_workers
        < 1
    ):

        raise ValueError(
            "--max-workers must be >= 1"
        )

    api_key = (
        os.environ.get(
            "OPENROUTER_API_KEY"
        )
    )

    if not api_key:

        raise RuntimeError(
            "export "
            "OPENROUTER_API_KEY='...'"
        )

    ensure_dir(
        args.output_dir
    )

    # =========================================================================
    # CONFIG
    # =========================================================================

    print(
        "=" * 112
    )

    print(
        "CONFIG"
    )

    print(
        "=" * 112
    )

    print(
        f"num_examples       = "
        f"{args.num_examples}"
    )

    print(
        f"seed               = "
        f"{args.seed}"
    )

    print(
        f"eval_set           = "
        f"{args.eval_set}"
    )

    print(
        f"top_k              = "
        f"{args.top_k}"
    )

    print(
        f"c6_overlap         = "
        f"size=6 "
        f"stride="
        f"{args.c6_overlap_stride}"
    )

    print(
        f"max_workers        = "
        f"{args.max_workers}"
    )

    print(
        f"BGE local          = "
        f"{args.bge_model}"
    )

    print(
        f"OpenRouter model   = "
        f"{args.model}"
    )

    print(
        "self_router        = "
        "DISABLED"
    )

    # =========================================================================
    # DATA
    # =========================================================================

    download_locomo(
        args.dataset
    )

    data = load_locomo(
        args.dataset
    )

    all_examples = (
        build_qa_examples(
            data
        )
    )

    if args.eval_set:

        examples = (
            load_selected_eval_set(
                args.eval_set,
                all_examples,
            )
        )

        print(
            f"[eval set] "
            f"loaded previous set: "
            f"{len(examples)}"
        )

    else:

        examples = (
            stratified_sample(
                all_examples,
                args.num_examples,
                args.seed,
            )
        )

    print(
        "\n[EVAL SET]"
    )

    print(
        f"n="
        f"{len(examples)}"
    )

    print(
        "categories="
        + str(
            dict(
                Counter(
                    str(
                        x[
                            "category"
                        ]
                    )
                    for x in examples
                )
            )
        )
    )

    # =========================================================================
    # MODEL
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
    # LOOP
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
            + "=" * 112
        )

        print(
            f"[{idx:03d}/"
            f"{len(examples):03d}] "
            f"sample="
            f"{sample_id} "
            f"qa="
            f"{qa_index} "
            f"category="
            f"{category}"
        )

        print(
            "=" * 112
        )

        print(
            f"Q    : "
            f"{question}"
        )

        print(
            f"GOLD : "
            f"{gold}"
        )

        # ---------------------------------------------------------------------
        # Build docs once
        # ---------------------------------------------------------------------

        if (
            sample_id
            not in doc_cache
        ):

            turns = (
                flatten_conversation(
                    sample
                )
            )

            doc_cache[
                sample_id
            ] = {

                "c6":
                    build_c6_overlap(
                        turns,
                        stride=(
                            args.c6_overlap_stride
                        ),
                    ),

                "summary":
                    build_summary_docs(
                        sample
                    ),
            }

            print(
                "\n[INDEX SIZE]"
            )

            print(
                f"  c6_overlap      "
                f"{len(doc_cache[sample_id]['c6'])}"
            )

            print(
                f"  summary         "
                f"{len(doc_cache[sample_id]['summary'])}"
            )

        docs = (
            doc_cache[
                sample_id
            ]
        )

        # ---------------------------------------------------------------------
        # Retrieval
        # ---------------------------------------------------------------------

        print(
            "\n[RETRIEVAL]"
        )

        t0 = time.time()

        c6_ret = (
            retriever.retrieve(
                query=question,
                docs=(
                    docs["c6"]
                ),
                top_k=(
                    args.top_k
                ),
            )
        )

        c6_hit = (
            raw_evidence_hit(
                c6_ret,
                evidence,
            )
        )

        print(
            f"  c6_overlap      "
            f"hit@{args.top_k}="
            f"{c6_hit} "
            f"time="
            f"{time.time()-t0:.2f}s"
        )

        for item in c6_ret:

            print(
                f"    "
                f"#{item['rank']} "
                f"{item['score']:.4f} "
                f"{shorten(item['text'], 110)}"
            )

        t0 = (
            time.time()
        )

        summary_ret = (
            retriever.retrieve(
                query=question,
                docs=(
                    docs[
                        "summary"
                    ]
                ),
                top_k=(
                    args.top_k
                ),
            )
        )

        summary_hit = (
            summary_evidence_hit(
                summary_ret,
                evidence,
            )
        )

        print(
            f"  summary         "
            f"hit@{args.top_k}="
            f"{summary_hit} "
            f"time="
            f"{time.time()-t0:.2f}s"
        )

        for item in (
            summary_ret
        ):

            print(
                f"    "
                f"#{item['rank']} "
                f"{item['score']:.4f} "
                f"{shorten(item['text'], 110)}"
            )

        # ---------------------------------------------------------------------
        # Final QA
        # NO SELF ROUTER
        # ---------------------------------------------------------------------

        print(
            "\n[FINAL QA - NO SELF ROUTER]"
        )

        predictions = {}
        input_chars = {}

        with ThreadPoolExecutor(
            max_workers=min(
                args.max_workers,
                2,
            )
        ) as executor:

            future_map = {

                executor.submit(
                    answer_question,
                    client,
                    args.model,
                    question,
                    c6_ret,
                ):
                    C6_METHOD,

                executor.submit(
                    answer_question,
                    client,
                    args.model,
                    question,
                    summary_ret,
                ):
                    SUMMARY_METHOD,
            }

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

                    pred, chars = (
                        future.result()
                    )

                except Exception as e:

                    print(
                        f"[QA ERROR] "
                        f"{method}: "
                        f"{e}"
                    )

                    pred = (
                        "__ERROR__"
                    )

                    chars = 0

                predictions[
                    method
                ] = pred

                input_chars[
                    method
                ] = chars

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        results = {}

        for method in (
            METHODS
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

            results[
                method
            ] = {

                "prediction":
                    pred,

                "em":
                    em,

                "strict_em":
                    strict,

                "f1":
                    f1,

                "input_chars":
                    input_chars[
                        method
                    ],
            }

            print(
                f"  {method:34s}"
                f"EM={em:.0f} "
                f"STRICT={strict:.0f} "
                f"F1={f1:.4f} "
                f"PRED="
                f"{shorten(pred, 110)}"
            )

        # ---------------------------------------------------------------------
        # Row
        # ---------------------------------------------------------------------

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

            "retrieval": {

                "c6_hit":
                    c6_hit,

                "summary_hit":
                    summary_hit,

                "c6_top5": [
                    {
                        "rank":
                            x[
                                "rank"
                            ],

                        "score":
                            x[
                                "score"
                            ],

                        "id":
                            x[
                                "id"
                            ],

                        "dia_ids":
                            x.get(
                                "dia_ids",
                                [],
                            ),
                    }
                    for x
                    in c6_ret
                ],

                "summary_top5": [
                    {
                        "rank":
                            x[
                                "rank"
                            ],

                        "score":
                            x[
                                "score"
                            ],

                        "id":
                            x[
                                "id"
                            ],

                        "session_ids":
                            x.get(
                                "session_ids",
                                [],
                            ),
                    }
                    for x
                    in summary_ret
                ],
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
        # Current
        # ---------------------------------------------------------------------

        print(
            "\n[CURRENT]"
        )

        for method in METHODS:

            r = (
                results[
                    method
                ]
            )

            mark = (
                "✓"
                if (
                    r["em"]
                    == 1
                )
                else "✗"
            )

            print(
                f"  {method:34s}"
                f"{mark} "
                f"EM={r['em']:.0f} "
                f"STRICT="
                f"{r['strict_em']:.0f} "
                f"F1="
                f"{r['f1']:.4f}"
            )

        print(
            f"\n  elapsed="
            f"{row['elapsed_sec']:.2f}s"
        )

        # ---------------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------------

        save_all(
            rows,
            args.output_dir,
        )

        # ---------------------------------------------------------------------
        # Cumulative
        # ---------------------------------------------------------------------

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
        + "#" * 112
    )

    print(
        "FINAL RESULT"
    )

    print(
        "#" * 112
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


if __name__ == "__main__":
    main()
