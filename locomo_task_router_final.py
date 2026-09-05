#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoCoMo Task-Routed Retrieval Experiment

FLOW
====
Query
  -> Binary Task Router
       0 = C6 overlap raw chunks
       1 = Session summaries
  -> Search ONLY selected representation with BGE-M3 Top-5
  -> Final QA
  -> EM / STRICT EM / F1

LOCAL
=====
BAAI/bge-m3

OPENROUTER
==========
qwen/qwen3.5-35b-a3b

FAIR COMPARISON
===============
Use the same eval set as previous experiments when possible:
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

import numpy as np
import requests


# =============================================================================
# CONFIG
# =============================================================================

LOCOMO_URL = (
    "https://raw.githubusercontent.com/"
    "snap-research/locomo/main/data/locomo10.json"
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

DEFAULT_BGE_MODEL = "BAAI/bge-m3"
DEFAULT_MODEL = "qwen/qwen3.5-35b-a3b"

METHOD = "task_router_c6_vs_summary"


# =============================================================================
# UTILS
# =============================================================================

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def shorten(text, n=120):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= n else text[:n] + "..."


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

    return " ".join(text.split())


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
        print(f"[dataset exists] {path}")
        return

    print("=" * 100)
    print("DOWNLOAD LOCOMO")
    print("=" * 100)

    r = requests.get(
        LOCOMO_URL,
        timeout=120,
    )

    r.raise_for_status()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_bytes(
        r.content
    )

    print(f"[saved] {path}")


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
    conv = sample["conversation"]

    turns = []

    for sn in get_session_numbers(conv):
        sid = f"session_{sn}"

        date = conv.get(
            f"session_{sn}_date_time"
        )

        for turn in conv[sid]:
            parsed = parse_turn(turn)

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
# C6 OVERLAP
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
# SESSION SUMMARIES
# =============================================================================

def build_summary_docs(sample):
    conv = sample[
        "conversation"
    ]

    summaries = sample.get(
        "session_summary",
        {},
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
# BGE-M3
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

            results.append(item)

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
                r = (
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

                if r.status_code == 429:
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

                r.raise_for_status()

                obj = r.json()

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
                    f"[OpenRouter ERROR] "
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
# TASK ROUTER
# =============================================================================

TASK_ROUTER_SYSTEM = """
You are a binary memory retrieval router.

Choose which memory representation should be searched for the question.

Output exactly ONE character:

0 = raw conversation chunks
1 = session summaries

Choose 0 when the question likely requires:
- an exact fact
- precise wording
- a name, object, attribute, number, date, place, or list
- local conversational detail
- evidence from specific dialogue turns

Choose 1 when the question likely requires:
- a broad session-level fact
- a high-level event
- a general activity or topic
- information likely preserved in a session summary

CRITICAL:
- Output ONLY 0 or 1.
- No explanation.
- No punctuation.
- No extra whitespace.
""".strip()


def parse_binary(output):
    output = str(
        output
    ).strip()

    if output == "0":
        return 0

    if output == "1":
        return 1

    m = re.search(
        r"[01]",
        output,
    )

    if m:
        return int(
            m.group(0)
        )

    # safer default for factual QA
    return 0


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

    raw_output = client.chat(
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
        raw_output
    )

    return (
        label,
        raw_output,
    )


# =============================================================================
# SAME QA PROMPT
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


def format_docs(docs):
    if not docs:
        return "NO RETRIEVED MEMORY"

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
# QA EXAMPLES
# =============================================================================

def build_qa_examples(data):
    examples = []

    for sample in data:
        sample_id = sample.get(
            "sample_id",
            "unknown",
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
            str(
                x["category"]
            )
        ].append(x)

    for g in groups.values():
        rng.shuffle(g)

    cats = sorted(
        groups.keys()
    )

    base = (
        n // len(cats)
    )

    rem = (
        n % len(cats)
    )

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

        for x
        in all_examples
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
# AGGREGATE
# =============================================================================

def aggregate(rows):
    return {
        "n":
            len(rows),

        "em":
            safe_mean(
                [
                    x["em"]
                    for x in rows
                ]
            ),

        "strict_em":
            safe_mean(
                [
                    x["strict_em"]
                    for x in rows
                ]
            ),

        "f1":
            safe_mean(
                [
                    x["f1"]
                    for x in rows
                ]
            ),

        "avg_input_chars":
            safe_mean(
                [
                    x["input_chars"]
                    for x in rows
                ]
            ),

        "retrieval_hit":
            safe_mean(
                [
                    x["retrieval_hit"]
                    for x in rows
                    if x["retrieval_hit"]
                    is not None
                ]
            ),
    }


def print_cumulative(
    rows,
    current,
    total,
):
    s = aggregate(
        rows
    )

    print(
        "\n"
        + "=" * 115
    )

    print(
        f"CUMULATIVE RESULT "
        f"[{current}/{total}]"
    )

    print(
        "=" * 115
    )

    print(
        f"{'METHOD':32s}"
        f"{'N':>6s}"
        f"{'EM':>10s}"
        f"{'STRICT':>10s}"
        f"{'F1':>10s}"
        f"{'HIT@5':>10s}"
        f"{'AVG_CHARS':>14s}"
    )

    print(
        "-" * 115
    )

    print(
        f"{METHOD:32s}"
        f"{s['n']:6d}"
        f"{s['em']:10.4f}"
        f"{s['strict_em']:10.4f}"
        f"{s['f1']:10.4f}"
        f"{s['retrieval_hit']:10.4f}"
        f"{s['avg_input_chars']:14.0f}"
    )

    print(
        "-" * 115
    )

    route_counter = Counter(
        row["route"]
        for row in rows
    )

    n = len(rows)

    print(
        "\n[TASK ROUTER]"
    )

    print(
        f"  raw/c6 (0) : "
        f"{route_counter.get(0,0)} "
        f"({route_counter.get(0,0)/n:.1%})"
    )

    print(
        f"  summary (1): "
        f"{route_counter.get(1,0)} "
        f"({route_counter.get(1,0)/n:.1%})"
    )

    # route-specific downstream performance
    for route, name in [
        (0, "raw/c6"),
        (1, "summary"),
    ]:
        subset = [
            x
            for x in rows
            if x["route"] == route
        ]

        if not subset:
            continue

        print(
            f"\n[ROUTE={route} {name}]"
        )

        print(
            f"  n={len(subset)} "
            f"EM={safe_mean([x['em'] for x in subset]):.4f} "
            f"STRICT={safe_mean([x['strict_em'] for x in subset]):.4f} "
            f"F1={safe_mean([x['f1'] for x in subset]):.4f}"
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

    s = aggregate(
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
                "retrieval_hit_at_5",
                "avg_input_chars",
            ]
        )

        w.writerow(
            [
                METHOD,
                s["n"],
                s["em"],
                s["strict_em"],
                s["f1"],
                s["retrieval_hit"],
                s["avg_input_chars"],
            ]
        )

    router_path = (
        Path(output_dir)
        / "router_choices.csv"
    )

    with open(
        router_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)

        w.writerow(
            [
                "sample_id",
                "qa_index",
                "category",
                "question",
                "route",
                "route_name",
                "router_raw_output",
                "em",
                "strict_em",
                "f1",
                "retrieval_hit",
            ]
        )

        for row in rows:
            w.writerow(
                [
                    row["sample_id"],
                    row["qa_index"],
                    row["category"],
                    row["question"],
                    row["route"],
                    row["route_name"],
                    row["router_raw_output"],
                    row["em"],
                    row["strict_em"],
                    row["f1"],
                    row["retrieval_hit"],
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
        default="./results_locomo_task_router_final",
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
        "--bge-model",
        default=DEFAULT_BGE_MODEL,
    )

    parser.add_argument(
        "--bge-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--router-model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--qa-model",
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

    print("=" * 115)
    print("CONFIG")
    print("=" * 115)

    print(
        f"num_examples        = {args.num_examples}"
    )

    print(
        f"seed                = {args.seed}"
    )

    print(
        f"eval_set            = {args.eval_set}"
    )

    print(
        f"top_k               = {args.top_k}"
    )

    print(
        f"c6 overlap          = size=6 stride={args.c6_overlap_stride}"
    )

    print(
        f"router model        = {args.router_model}"
    )

    print(
        f"qa model            = {args.qa_model}"
    )

    print(
        "routing             = query-only binary"
    )

    print(
        "0                   = C6 overlap"
    )

    print(
        "1                   = session summary"
    )

    # -------------------------------------------------------------------------
    # data
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
            f"[eval] loaded previous set: "
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
    # models
    # -------------------------------------------------------------------------

    retriever = BGEM3Retriever(
        args.bge_model,
        batch_size=args.bge_batch_size,
    )

    client = OpenRouterClient(
        api_key
    )

    doc_cache = {}
    rows = []

    # =========================================================================
    # EVAL
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
            + "=" * 115
        )

        print(
            f"[{idx:03d}/{len(examples):03d}] "
            f"sample={sample_id} "
            f"qa={qa_index} "
            f"category={category}"
        )

        print(
            "=" * 115
        )

        print(
            f"Q    : {question}"
        )

        print(
            f"GOLD : {gold}"
        )

        # ---------------------------------------------------------------------
        # build docs
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

        # ---------------------------------------------------------------------
        # task router BEFORE retrieval
        # ---------------------------------------------------------------------

        route, router_raw = task_route(
            client,
            args.router_model,
            question,
        )

        route_name = (
            "summary"
            if route == 1
            else "c6_overlap"
        )

        print(
            "\n[TASK ROUTER]"
        )

        print(
            f"  raw_output={router_raw!r}"
        )

        print(
            f"  route={route} "
            f"({route_name})"
        )

        # ---------------------------------------------------------------------
        # ONLY search selected representation
        # ---------------------------------------------------------------------

        if route == 0:
            retrieved = retriever.retrieve(
                question,
                doc_cache[sample_id]["c6"],
                top_k=args.top_k,
            )

            retrieval_hit = raw_evidence_hit(
                retrieved,
                evidence,
            )

        else:
            retrieved = retriever.retrieve(
                question,
                doc_cache[sample_id]["summary"],
                top_k=args.top_k,
            )

            retrieval_hit = summary_evidence_hit(
                retrieved,
                evidence,
            )

        print(
            "\n[RETRIEVAL]"
        )

        print(
            f"  source={route_name}"
        )

        print(
            f"  hit@{args.top_k}={retrieval_hit}"
        )

        for doc in retrieved:
            print(
                f"    "
                f"#{doc['rank']} "
                f"{doc['score']:.4f} "
                f"{shorten(doc['text'], 110)}"
            )

        # ---------------------------------------------------------------------
        # final QA
        # ---------------------------------------------------------------------

        pred, input_chars = answer_question(
            client,
            args.qa_model,
            question,
            retrieved,
        )

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

        print(
            "\n[FINAL QA]"
        )

        print(
            f"  PRED   : {pred}"
        )

        print(
            f"  EM     : {em:.0f}"
        )

        print(
            f"  STRICT : {strict:.0f}"
        )

        print(
            f"  F1     : {f1:.4f}"
        )

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

            "route":
                route,

            "route_name":
                route_name,

            "router_raw_output":
                router_raw,

            "retrieval_hit":
                retrieval_hit,

            "prediction":
                pred,

            "em":
                em,

            "strict_em":
                strict,

            "f1":
                f1,

            "input_chars":
                input_chars,

            "elapsed_sec":
                time.time() - start,
        }

        rows.append(
            row
        )

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
        + "#" * 115
    )

    print(
        "FINAL RESULT"
    )

    print(
        "#" * 115
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
        "  router_choices.csv"
    )


if __name__ == "__main__":
    main()
