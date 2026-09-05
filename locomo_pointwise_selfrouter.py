#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoCoMo pointwise self-router experiment

LOCAL
=====
BAAI/bge-m3

OPENROUTER
==========
- pointwise relevance judge: output ONLY 0 or 1
- task router: output ONLY 0 or 1
- final QA

METHODS
=======
1. c6_overlap_selfrouter
   C6-overlap Top-5
   -> each doc judged 0/1
   -> keep only relevant docs
   -> answer

2. summary_selfrouter
   Session-summary Top-5
   -> each summary judged 0/1
   -> keep only relevant docs
   -> answer

3. taskrouter_selfrouter
   Query-only task router:
       0 = use C6 overlap
       1 = use summary
   -> selected representation Top-5
   -> each doc judged 0/1
   -> answer from relevant docs only

4. both_selfrouter
   C6-overlap Top-5 + Summary Top-5
   -> 10 docs total
   -> each doc judged 0/1
   -> keep relevant docs
   -> answer

METRICS
=======
- relaxed EM
- strict EM
- token F1
- retrieval evidence hit@5
- post-self-router evidence retained
- avg retrieved docs
- avg kept docs
- self-router keep rate
- task-router distribution
- avg context chars

IMPORTANT
=========
For fair comparison with previous run:
- same LoCoMo data
- same seed
- same stratified sampling logic

Even safer:
--eval-set ./results_locomo_final/selected_eval_set.json
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

C6 = "c6_overlap_selfrouter"
SUMMARY = "summary_selfrouter"
TASK = "taskrouter_selfrouter"
BOTH = "both_selfrouter"

METHODS = [
    C6,
    SUMMARY,
    TASK,
    BOTH,
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
    return sum(xs) / len(xs) if xs else 0.0


def shorten(text, n=130):
    text = re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()

    return text if len(text) <= n else text[:n] + "..."


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
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()

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

    precision = common / len(p)
    recall = common / len(g)

    return (
        2 * precision * recall
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
        f"[LoCoMo] conversations={len(data)}"
    )

    return data


# =============================================================================
# LOCOMO PARSING
# =============================================================================

def get_session_numbers(conv):
    nums = []

    for key in conv.keys():
        m = re.fullmatch(
            r"session_(\d+)",
            key,
        )

        if m:
            nums.append(
                int(m.group(1))
            )

    return sorted(nums)


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

    for sn in get_session_numbers(conv):
        sid = f"session_{sn}"

        date = conv.get(
            f"session_{sn}_date_time"
        )

        for turn in conv[sid]:
            parsed = parse_turn(
                turn
            )

            turns.append(
                {
                    "session_num":
                        sn,
                    "session_id":
                        sid,
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
# CHUNKING
# =============================================================================

def make_chunks(
    turns,
    chunk_size,
    stride,
):
    docs = []

    start = 0
    chunk_idx = 0

    while start < len(turns):
        sub = turns[
            start:
            start + chunk_size
        ]

        if not sub:
            break

        lines = []

        for t in sub:
            meta = []

            if t["date"]:
                meta.append(
                    str(t["date"])
                )

            meta.append(
                t["session_id"]
            )

            lines.append(
                "["
                + " | ".join(meta)
                + "] "
                + t["text"]
            )

        docs.append(
            {
                "id":
                    f"chunk_{chunk_idx}",

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
                    start + len(sub) - 1,
            }
        )

        chunk_idx += 1
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
# SUMMARY DOCS
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

    for sn in get_session_numbers(conv):
        key = (
            f"session_{sn}_summary"
        )

        if key not in summaries:
            continue

        summary = summaries[key]

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

        sid = f"session_{sn}"

        date = conv.get(
            f"session_{sn}_date_time"
        )

        docs.append(
            {
                "id":
                    key,

                "text":
                    (
                        f"SESSION: {sid}\n"
                        f"DATE: {date}\n"
                        f"SUMMARY:\n"
                        f"{summary}"
                    ),

                "session_ids": [
                    sid
                ],

                "dia_ids":
                    [],
            }
        )

    return docs


# =============================================================================
# BGE
# =============================================================================

class BGEM3Retriever:

    def __init__(
        self,
        model_name,
        batch_size=32,
        max_length=8192,
    ):
        from FlagEmbedding import BGEM3FlagModel

        print("=" * 100)
        print("LOAD LOCAL BGE-M3")
        print("=" * 100)

        print(
            f"model={model_name}"
        )

        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=True,
        )

        self.batch_size = batch_size
        self.max_length = max_length

    def encode(self, texts):
        output = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )

        vecs = np.asarray(
            output["dense_vecs"],
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            vecs,
            axis=1,
            keepdims=True,
        )

        return (
            vecs
            / np.maximum(
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

        qvec = self.encode(
            [query]
        )[0]

        dvecs = self.encode(
            [
                d["text"]
                for d in docs
            ]
        )

        scores = (
            dvecs @ qvec
        )

        order = (
            np.argsort(-scores)
            [:top_k]
        )

        results = []

        for rank, idx in enumerate(
            order,
            1,
        ):
            item = dict(
                docs[int(idx)]
            )

            item["rank"] = rank
            item["score"] = float(
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
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.local = threading.local()

    def _session(self):
        if not hasattr(
            self.local,
            "session",
        ):
            s = requests.Session()

            s.headers.update(
                {
                    "Authorization":
                        f"Bearer {self.api_key}",
                    "Content-Type":
                        "application/json",
                }
            )

            self.local.session = s

        return self.local.session

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
                "enabled": False
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

                if response.status_code == 429:
                    wait = min(
                        2 ** attempt,
                        10,
                    )

                    print(
                        f"[429] sleep={wait}",
                        flush=True,
                    )

                    time.sleep(wait)
                    continue

                response.raise_for_status()

                obj = response.json()

                content = (
                    obj["choices"][0]
                    ["message"]
                    .get(
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
                    f"[OR ERROR] "
                    f"{type(e).__name__}: {e}",
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
            str(last_error)
        )


# =============================================================================
# POINTWISE SELF ROUTER
# =============================================================================

SELF_ROUTER_SYSTEM = """
You are a binary relevance classifier.

Given a question and ONE retrieved memory document, decide whether this
document contains information that is useful for answering the question.

Output exactly ONE character:

1 = relevant
0 = irrelevant

Rules:
- Output ONLY 0 or 1.
- No explanation.
- No punctuation.
- No whitespace around the digit.
- Mark 1 if the document directly contains the answer OR provides
  necessary supporting context for answering the question.
- Mark 0 if it is unrelated, merely topically similar, or does not help
  answer the specific question.
""".strip()


def parse_binary(output):
    output = str(output).strip()

    m = re.search(
        r"[01]",
        output,
    )

    if not m:
        return 0

    return int(
        m.group(0)
    )


def judge_one_doc(
    client,
    model,
    question,
    doc,
):
    prompt = f"""
QUESTION:
{question}

DOCUMENT:
{doc['text']}

LABEL:
""".strip()

    output = client.chat(
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

        max_tokens=3,
        temperature=0.0,
    )

    label = parse_binary(
        output
    )

    return {
        "label":
            label,
        "raw_output":
            output,
    }


def pointwise_filter(
    client,
    model,
    question,
    docs,
    max_workers,
):
    """
    Judge every retrieved document independently.

    This is the actual self-router.
    """

    judged = [
        None
        for _ in docs
    ]

    workers = min(
        max_workers,
        max(
            1,
            len(docs),
        ),
    )

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        future_map = {}

        for idx, doc in enumerate(
            docs
        ):
            future = executor.submit(
                judge_one_doc,
                client,
                model,
                question,
                doc,
            )

            future_map[
                future
            ] = idx

        for future in as_completed(
            future_map
        ):
            idx = future_map[
                future
            ]

            try:
                judged[
                    idx
                ] = future.result()

            except Exception as e:
                print(
                    f"[SELF ROUTER ERROR] "
                    f"doc={idx} "
                    f"{e}",
                    flush=True,
                )

                judged[idx] = {
                    "label":
                        0,
                    "raw_output":
                        "__ERROR__",
                }

    kept = []

    detailed = []

    for doc, result in zip(
        docs,
        judged,
    ):
        item = dict(
            doc
        )

        item[
            "self_router_label"
        ] = (
            result["label"]
        )

        item[
            "self_router_raw"
        ] = (
            result["raw_output"]
        )

        detailed.append(
            item
        )

        if result[
            "label"
        ] == 1:
            kept.append(
                item
            )

    return kept, detailed


# =============================================================================
# TASK ROUTER
# =============================================================================

TASK_ROUTER_SYSTEM = """
You are a binary task router.

Decide which memory representation should be searched for this question.

Output exactly ONE character:

0 = raw conversation chunks
1 = session summaries

Use 0 when:
- exact details matter
- wording matters
- local dialogue context matters
- lists, names, attributes, dates, or specific facts are requested

Use 1 when:
- the question asks for a broad session-level fact
- a compressed summary is likely sufficient

Output ONLY 0 or 1.
No explanation.
No punctuation.
""".strip()


def task_route(
    client,
    model,
    question,
):
    prompt = f"""
QUESTION:
{question}

ROUTE:
""".strip()

    output = client.chat(
        model=model,

        messages=[
            {
                "role":
                    "system",
                "content":
                    TASK_ROUTER_SYSTEM,
            },
            {
                "role":
                    "user",
                "content":
                    prompt,
            },
        ],

        max_tokens=3,
        temperature=0.0,
    )

    label = parse_binary(
        output
    )

    return (
        label,
        output,
    )


# =============================================================================
# FINAL QA
# =============================================================================

QA_SYSTEM = """
You answer short-answer questions using only the supplied memory.

Your output is evaluated with exact-match style metrics.

Rules:
- Return ONLY the shortest answer span.
- Do not write a full explanatory sentence.
- Do not repeat the question.
- Do not say "The answer is".
- Do not explain.
- Preserve wording from the memory whenever possible.
- For lists, return only the requested list.
- If the relevant memory does not contain the answer, output exactly:
  unknown
""".strip()


def format_docs(
    docs,
):
    if not docs:
        return "NO RELEVANT MEMORY"

    blocks = []

    for idx, doc in enumerate(
        docs,
        1,
    ):
        blocks.append(
            f"[MEMORY {idx}]\n"
            f"{doc['text']}"
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
    context = format_docs(
        docs
    )

    prompt = f"""
MEMORY:
{context}

QUESTION:
{question}

SHORT ANSWER:
""".strip()

    pred = client.chat(
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

    return (
        pred,
        len(context),
    )


# =============================================================================
# RETRIEVAL EVAL
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

    found = set()

    for doc in docs:
        found.update(
            doc.get(
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
    docs,
    evidence,
):
    gold_sessions = set()

    for eid in (
        evidence or []
    ):
        m = re.match(
            r"D(\d+):",
            str(eid),
        )

        if m:
            gold_sessions.add(
                f"session_{int(m.group(1))}"
            )

    if not gold_sessions:
        return None

    found = set()

    for doc in docs:
        found.update(
            doc.get(
                "session_ids",
                [],
            )
        )

    return int(
        bool(
            found
            & gold_sessions
        )
    )


# =============================================================================
# QA SET
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
                gold = qa.get(
                    "adversarial_answer"
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
            str(x["category"])
        ].append(x)

    for g in groups.values():
        rng.shuffle(
            g
        )

    cats = sorted(
        groups.keys()
    )

    base = n // len(cats)
    rem = n % len(cats)

    selected = []

    for idx, cat in enumerate(
        cats
    ):
        want = (
            base
            + (
                1
                if idx < rem
                else 0
            )
        )

        selected.extend(
            groups[
                cat
            ][:want]
        )

    rng.shuffle(
        selected
    )

    return selected[:n]


def load_selected_eval_set(
    path,
    all_examples,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        selected = json.load(f)

    lookup = {
        (
            x["sample_id"],
            x["qa_index"],
        ):
            x
        for x in all_examples
    }

    out = []

    for item in selected:
        key = (
            item["sample_id"],
            item["qa_index"],
        )

        if key not in lookup:
            raise KeyError(
                f"Eval example not found: {key}"
            )

        out.append(
            lookup[key]
        )

    return out


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

            "avg_kept":
                safe_mean(
                    [
                        x["kept_docs"]
                        for x in vals
                    ]
                ),

            "avg_input_chars":
                safe_mean(
                    [
                        x["input_chars"]
                        for x in vals
                    ]
                ),
        }

    return stats


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
        + "=" * 125
    )

    print(
        f"CUMULATIVE RESULT "
        f"[{current}/{total}]"
    )

    print(
        "=" * 125
    )

    print(
        f"{'METHOD':30s}"
        f"{'N':>5s}"
        f"{'EM':>9s}"
        f"{'STRICT':>10s}"
        f"{'F1':>10s}"
        f"{'KEPT':>9s}"
        f"{'AVG_CHARS':>13s}"
    )

    print(
        "-" * 125
    )

    for method in METHODS:
        s = stats[
            method
        ]

        print(
            f"{method:30s}"
            f"{s['n']:5d}"
            f"{s['em']:9.4f}"
            f"{s['strict_em']:10.4f}"
            f"{s['f1']:10.4f}"
            f"{s['avg_kept']:9.2f}"
            f"{s['avg_input_chars']:13.0f}"
        )

    print(
        "-" * 125
    )

    task_dist = Counter(
        row[
            "task_router"
        ][
            "route"
        ]
        for row in rows
    )

    print(
        "\n[TASK ROUTER]"
    )

    print(
        f"  raw(0)     : "
        f"{task_dist.get(0,0)} "
        f"({task_dist.get(0,0)/len(rows):.1%})"
    )

    print(
        f"  summary(1) : "
        f"{task_dist.get(1,0)} "
        f"({task_dist.get(1,0)/len(rows):.1%})"
    )


# =============================================================================
# SAVE
# =============================================================================

def save_all(
    rows,
    output_dir,
):
    detail_path = (
        Path(output_dir)
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
        Path(output_dir)
        / "summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)

        w.writerow(
            [
                "method",
                "n",
                "em",
                "strict_em",
                "f1",
                "avg_kept_docs",
                "avg_input_chars",
            ]
        )

        for method in METHODS:
            s = stats[
                method
            ]

            w.writerow(
                [
                    method,
                    s["n"],
                    s["em"],
                    s["strict_em"],
                    s["f1"],
                    s["avg_kept"],
                    s["avg_input_chars"],
                ]
            )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="./locomo10.json",
    )

    parser.add_argument(
        "--output-dir",
        default="./results_locomo_pointwise_selfrouter",
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
            "Optional previous selected_eval_set.json. "
            "Use this for exact same 50 examples."
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
        default=10,
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
        "--model",
        default=DEFAULT_MODEL,
    )

    args = parser.parse_args()

    api_key = os.environ.get(
        "OPENROUTER_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "export OPENROUTER_API_KEY='...'"
        )

    ensure_dir(
        args.output_dir
    )

    print("=" * 120)
    print("CONFIG")
    print("=" * 120)

    print(
        f"num_examples      = {args.num_examples}"
    )

    print(
        f"seed              = {args.seed}"
    )

    print(
        f"eval_set          = {args.eval_set}"
    )

    print(
        f"top_k             = {args.top_k}"
    )

    print(
        f"max_workers       = {args.max_workers}"
    )

    print(
        f"c6_stride         = {args.c6_overlap_stride}"
    )

    print(
        f"model             = {args.model}"
    )

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    download_locomo(
        args.dataset
    )

    data = load_locomo(
        args.dataset
    )

    all_examples = build_qa_examples(
        data
    )

    if args.eval_set:
        examples = load_selected_eval_set(
            args.eval_set,
            all_examples,
        )

        print(
            f"[eval] loaded exact previous set: "
            f"{len(examples)}"
        )

    else:
        examples = stratified_sample(
            all_examples,
            args.num_examples,
            args.seed,
        )

    print(
        "[categories]",
        dict(
            Counter(
                str(x["category"])
                for x in examples
            )
        ),
    )

    # -------------------------------------------------------------------------
    # MODELS
    # -------------------------------------------------------------------------

    retriever = BGEM3Retriever(
        args.bge_model,
        batch_size=args.bge_batch_size,
    )

    client = OpenRouterClient(
        api_key
    )

    # -------------------------------------------------------------------------
    # DOC CACHE
    # -------------------------------------------------------------------------

    doc_cache = {}

    rows = []

    # =========================================================================
    # LOOP
    # =========================================================================

    for idx, ex in enumerate(
        examples,
        1,
    ):
        start = time.time()

        sample = ex["sample"]
        sample_id = ex["sample_id"]
        qa_index = ex["qa_index"]
        question = ex["question"]
        gold = ex["answer"]
        category = ex["category"]
        evidence = ex["evidence"]

        print(
            "\n\n"
            + "=" * 120
        )

        print(
            f"[{idx:03d}/{len(examples):03d}] "
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

        if sample_id not in doc_cache:
            turns = flatten_conversation(
                sample
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
                f"  c6_overlap        "
                f"{len(doc_cache[sample_id]['c6'])}"
            )

            print(
                f"  session_summary   "
                f"{len(doc_cache[sample_id]['summary'])}"
            )

        docs = doc_cache[
            sample_id
        ]

        # ---------------------------------------------------------------------
        # retrieve both
        # ---------------------------------------------------------------------

        print(
            "\n[RETRIEVAL]"
        )

        c6_ret = retriever.retrieve(
            question,
            docs["c6"],
            top_k=args.top_k,
        )

        summary_ret = retriever.retrieve(
            question,
            docs["summary"],
            top_k=args.top_k,
        )

        c6_hit = raw_evidence_hit(
            c6_ret,
            evidence,
        )

        sum_hit = summary_evidence_hit(
            summary_ret,
            evidence,
        )

        print(
            f"  c6_overlap      "
            f"hit@5={c6_hit}"
        )

        print(
            f"  summary         "
            f"hit@5={sum_hit}"
        )

        # ---------------------------------------------------------------------
        # task route
        # ---------------------------------------------------------------------

        task_label, task_raw = task_route(
            client,
            args.model,
            question,
        )

        task_source = (
            "summary"
            if task_label == 1
            else "c6"
        )

        print(
            "\n[TASK ROUTER]"
        )

        print(
            f"  output={task_raw!r}"
        )

        print(
            f"  route={task_label} "
            f"({'summary' if task_label else 'raw_c6'})"
        )

        # ---------------------------------------------------------------------
        # prepare four source sets
        # ---------------------------------------------------------------------

        source_sets = {
            C6:
                list(c6_ret),

            SUMMARY:
                list(summary_ret),

            TASK:
                (
                    list(summary_ret)
                    if task_label == 1
                    else list(c6_ret)
                ),

            BOTH:
                (
                    [
                        {
                            **x,
                            "source_type":
                                "c6",
                        }
                        for x in c6_ret
                    ]
                    +
                    [
                        {
                            **x,
                            "source_type":
                                "summary",
                        }
                        for x in summary_ret
                    ]
                ),
        }

        # ---------------------------------------------------------------------
        # pointwise self-router for each method
        # ---------------------------------------------------------------------

        filtered = {}
        judged = {}

        print(
            "\n[POINTWISE SELF ROUTER]"
        )

        for method in METHODS:
            kept, detail = pointwise_filter(
                client,
                args.model,
                question,
                source_sets[
                    method
                ],
                args.max_workers,
            )

            filtered[
                method
            ] = kept

            judged[
                method
            ] = detail

            labels = [
                x[
                    "self_router_label"
                ]
                for x in detail
            ]

            print(
                f"  {method:30s}"
                f"labels={labels} "
                f"kept={len(kept)}/{len(detail)}"
            )

        # ---------------------------------------------------------------------
        # final QA in parallel
        # ---------------------------------------------------------------------

        print(
            "\n[FINAL QA]"
        )

        predictions = {}
        input_chars = {}

        with ThreadPoolExecutor(
            max_workers=min(
                args.max_workers,
                len(METHODS),
            )
        ) as executor:

            future_map = {}

            for method in METHODS:
                future = executor.submit(
                    answer_question,
                    client,
                    args.model,
                    question,
                    filtered[
                        method
                    ],
                )

                future_map[
                    future
                ] = method

            for future in as_completed(
                future_map
            ):
                method = future_map[
                    future
                ]

                pred, chars = future.result()

                predictions[
                    method
                ] = pred

                input_chars[
                    method
                ] = chars

        # ---------------------------------------------------------------------
        # metrics
        # ---------------------------------------------------------------------

        results = {}

        for method in METHODS:
            pred = predictions[
                method
            ]

            em = relaxed_exact_match(
                pred,
                gold,
            )

            strict = strict_exact_match(
                pred,
                gold,
            )

            f1 = token_f1(
                pred,
                gold,
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

                "retrieved_docs":
                    len(
                        source_sets[
                            method
                        ]
                    ),

                "kept_docs":
                    len(
                        filtered[
                            method
                        ]
                    ),

                "input_chars":
                    input_chars[
                        method
                    ],
            }

            print(
                f"  {method:30s}"
                f"EM={em:.0f} "
                f"STRICT={strict:.0f} "
                f"F1={f1:.4f} "
                f"kept="
                f"{len(filtered[method])}/"
                f"{len(source_sets[method])} "
                f"PRED="
                f"{shorten(pred,100)}"
            )

        # ---------------------------------------------------------------------
        # post-filter evidence retained
        # ---------------------------------------------------------------------

        post_filter_hits = {
            C6:
                raw_evidence_hit(
                    filtered[
                        C6
                    ],
                    evidence,
                ),

            SUMMARY:
                summary_evidence_hit(
                    filtered[
                        SUMMARY
                    ],
                    evidence,
                ),

            TASK:
                (
                    summary_evidence_hit(
                        filtered[
                            TASK
                        ],
                        evidence,
                    )
                    if task_label == 1
                    else
                    raw_evidence_hit(
                        filtered[
                            TASK
                        ],
                        evidence,
                    )
                ),
        }

        # BOTH can contain mixed types.
        both_raw = [
            x
            for x in filtered[
                BOTH
            ]
            if x.get(
                "source_type"
            ) == "c6"
        ]

        both_sum = [
            x
            for x in filtered[
                BOTH
            ]
            if x.get(
                "source_type"
            ) == "summary"
        ]

        both_hit_raw = raw_evidence_hit(
            both_raw,
            evidence,
        )

        both_hit_sum = summary_evidence_hit(
            both_sum,
            evidence,
        )

        post_filter_hits[
            BOTH
        ] = int(
            bool(
                (both_hit_raw or 0)
                or
                (both_hit_sum or 0)
            )
        )

        # ---------------------------------------------------------------------
        # row
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

            "task_router": {
                "route":
                    task_label,
                "source":
                    task_source,
                "raw_output":
                    task_raw,
            },

            "retrieval": {
                "c6_hit_at_5":
                    c6_hit,

                "summary_hit_at_5":
                    sum_hit,

                "post_selfrouter_hits":
                    post_filter_hits,
            },

            "results":
                results,

            "judged_docs": {
                method: [
                    {
                        "id":
                            x["id"],

                        "rank":
                            x.get(
                                "rank"
                            ),

                        "score":
                            x.get(
                                "score"
                            ),

                        "label":
                            x[
                                "self_router_label"
                            ],

                        "raw":
                            x[
                                "self_router_raw"
                            ],
                    }
                    for x in judged[
                        method
                    ]
                ]

                for method in METHODS
            },

            "elapsed_sec":
                time.time() - start,
        }

        rows.append(
            row
        )

        # ---------------------------------------------------------------------
        # print current
        # ---------------------------------------------------------------------

        print(
            "\n[CURRENT]"
        )

        for method in METHODS:
            r = results[
                method
            ]

            print(
                f"  {method:30s}"
                f"EM={r['em']:.0f} "
                f"STRICT={r['strict_em']:.0f} "
                f"F1={r['f1']:.4f} "
                f"kept={r['kept_docs']}"
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
        + "#" * 125
    )

    print(
        "FINAL RESULT"
    )

    print(
        "#" * 125
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
