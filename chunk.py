#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunked Memory Extraction Ablation
==================================

Goal
----
Evaluate whether:
1) typed memory helps:
   - untyped
   - joint_typed
   - sequential_typed

2) chunk granularity helps:
   - 2 chunks
   - 5 chunks

Pipeline
--------
For each LongMemEval example:

Dialogue
  -> split chronologically into K large chunks
  -> extract memory ONCE per chunk
  -> concatenate extracted memories
  -> final QA reasoning
  -> normalized Exact Match

Important
---------
- No LLM judge.
- Only normalized EM.
- No recursive summarization.
- Chunk 2 does NOT receive memory from Chunk 1.
- Each dialogue chunk is processed exactly once.
- The final QA model sees ONLY extracted memories,
  not the original conversation.

Dataset
-------
LongMemEval Oracle.

Why Oracle?
-----------
Oracle includes evidence sessions, reducing retrieval as a confound.
This experiment focuses on memory extraction / representation.

Install
-------
pip install openai huggingface_hub tqdm

Example
-------
export OPENROUTER_API_KEY="sk-or-v1-..."

python3 chunked_memory_ablation.py \
    --limit 50 \
    --workers 10 \
    --model google/gemini-2.5-flash \
    --chunks 2 5 \
    --methods untyped joint_typed sequential_typed \
    --memory-tokens 300 \
    --qa-tokens 64
"""

import os
import re
import json
import time
import random
import string
import argparse
import threading
from pathlib import Path
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import OpenAI
from huggingface_hub import hf_hub_download


# ============================================================
# CONSTANTS
# ============================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DATA_REPO = "xiaowu0162/longmemeval-cleaned"
DATA_FILE = "longmemeval_oracle.json"

DEFAULT_METHODS = [
    "untyped",
    "joint_typed",
    "sequential_typed",
]

DEFAULT_CHUNKS = [2, 5]

_thread_local = threading.local()


# ============================================================
# OPENROUTER
# ============================================================

def get_client():
    if not hasattr(_thread_local, "client"):

        api_key = os.environ.get("OPENROUTER_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set.\n\n"
                'export OPENROUTER_API_KEY="sk-or-v1-..."\n'
            )

        _thread_local.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )

    return _thread_local.client


def call_llm(
    model,
    system,
    user,
    max_tokens,
    temperature=0.0,
    retries=5,
):
    client = get_client()

    last_error = None

    for attempt in range(retries):

        try:
            response = client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": system,
                    },
                    {
                        "role": "user",
                        "content": user,
                    },
                ],
            )

            content = response.choices[0].message.content

            if content is None:
                content = ""

            return content.strip()

        except Exception as e:
            last_error = e

            sleep_sec = min(
                2 ** attempt + random.random(),
                20,
            )

            print(
                f"\n[retry {attempt+1}/{retries}] "
                f"{type(e).__name__}: {e}"
            )

            time.sleep(sleep_sec)

    raise RuntimeError(
        f"LLM failed: {last_error}"
    )


# ============================================================
# DATASET
# ============================================================

def download_dataset(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = data_dir / DATA_FILE

    if path.exists():
        print(
            f"[dataset] exists: {path}"
        )

        return path

    print(
        "[dataset] downloading..."
    )

    downloaded = hf_hub_download(
        repo_id=DATA_REPO,
        filename=DATA_FILE,
        repo_type="dataset",
        local_dir=str(data_dir),
    )

    return Path(downloaded)


def load_dataset(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    print(
        f"[dataset] examples={len(data)}"
    )

    return data


# ============================================================
# SAMPLE
# ============================================================

def balanced_sample(
    data,
    limit,
    seed=42,
    include_abstention=False,
):
    rnd = random.Random(seed)

    groups = defaultdict(list)

    for item in data:

        qid = str(
            item["question_id"]
        )

        if (
            not include_abstention
            and qid.endswith("_abs")
        ):
            continue

        qtype = item[
            "question_type"
        ]

        groups[qtype].append(
            item
        )

    for group in groups.values():
        rnd.shuffle(group)

    qtypes = sorted(
        groups.keys()
    )

    print(
        "\n[available question types]"
    )

    for qt in qtypes:
        print(
            f"  {qt:30s}: "
            f"{len(groups[qt])}"
        )

    selected = []

    idx = 0

    while len(selected) < limit:

        added = False

        for qt in qtypes:

            if idx < len(
                groups[qt]
            ):
                selected.append(
                    groups[qt][idx]
                )

                added = True

                if len(selected) >= limit:
                    break

        if not added:
            break

        idx += 1

    rnd.shuffle(selected)

    print(
        "\n[selected distribution]"
    )

    counts = Counter(
        x["question_type"]
        for x in selected
    )

    for qt, n in sorted(
        counts.items()
    ):
        print(
            f"  {qt:30s}: {n}"
        )

    print(
        f"  {'TOTAL':30s}: "
        f"{len(selected)}"
    )

    return selected


# ============================================================
# SESSION FORMATTING
# ============================================================

def format_session(
    session,
    date,
    session_idx,
):

    lines = [
        f"===== SESSION {session_idx} =====",
        f"DATE: {date}",
    ]

    for turn in session:

        role = str(
            turn.get(
                "role",
                "unknown",
            )
        ).upper()

        content = str(
            turn.get(
                "content",
                "",
            )
        ).strip()

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# CHUNKING
# ============================================================

def split_evenly(
    items,
    n_chunks,
):
    """
    Chronological contiguous splitting.

    Example:
    len=10, chunks=3

    -> 4 / 3 / 3

    No session is duplicated.
    """

    n = len(items)

    if n == 0:
        return []

    n_chunks = min(
        n_chunks,
        n,
    )

    base = n // n_chunks
    remainder = n % n_chunks

    chunks = []

    start = 0

    for i in range(n_chunks):

        size = (
            base
            + (1 if i < remainder else 0)
        )

        end = start + size

        chunks.append(
            items[start:end]
        )

        start = end

    return chunks


def build_dialogue_chunks(
    item,
    n_chunks,
):
    """
    Split at SESSION boundaries.

    This is preferable to cutting arbitrary text/token positions because
    LongMemEval is session-structured and chronology matters.
    """

    sessions = item[
        "haystack_sessions"
    ]

    dates = item[
        "haystack_dates"
    ]

    session_items = []

    for idx, (
        session,
        date,
    ) in enumerate(
        zip(
            sessions,
            dates,
        ),
        start=1,
    ):

        session_items.append(
            {
                "session_idx": idx,
                "date": date,
                "session": session,
            }
        )

    split = split_evenly(
        session_items,
        n_chunks,
    )

    output = []

    for chunk_idx, chunk in enumerate(
        split,
        start=1,
    ):

        texts = []

        for x in chunk:

            texts.append(
                format_session(
                    x["session"],
                    x["date"],
                    x["session_idx"],
                )
            )

        output.append(
            "\n\n".join(
                texts
            )
        )

    return output


# ============================================================
# EXTRACTION PROMPTS
# ============================================================

UNTYPED_SYSTEM = """
You are a long-term memory extraction system.

Extract only information from the supplied dialogue chunk that may be
useful for answering future questions.

Important rules:

1. Use only information explicitly supported by the dialogue.
2. Do not guess or invent information.
3. Preserve names, dates, numbers, quantities, locations, relationships,
   preferences, actions, decisions, and changes over time.
4. Preserve temporal information when available.
5. If something changes over time, retain enough detail to distinguish
   the old state from the new state.
6. Prefer concise atomic memories rather than vague summaries.
7. Do not answer any future question.
8. Do not include irrelevant conversational filler.
""".strip()


def untyped_prompt(chunk):
    return f"""
Extract useful long-term memories from this dialogue chunk.

Return JSON only using exactly this schema:

{{
  "memories": [
    "atomic memory",
    "atomic memory"
  ]
}}

Rules:
- Each memory should be independently understandable.
- Preserve explicit dates when useful.
- Preserve important event and factual information.
- Do not categorize memories by type.
- Do not infer unsupported details.
- Return JSON only.

DIALOGUE CHUNK
==============

{chunk}
""".strip()


TYPED_SYSTEM = """
You are a precise long-term memory extraction system.

There are two kinds of memory.

EPISODIC MEMORY
----------------
A memory describing a particular event, action, interaction, experience,
decision, occurrence, or change situated in time.

Examples:
- The user traveled to Boston last Friday.
- The user bought a new laptop in March.
- The assistant recommended Restaurant A on June 3.
- The user changed jobs in 2025.

FACT MEMORY
-----------
A relatively durable semantic fact that remains useful outside a single
event.

Examples:
- The user prefers Italian food.
- The user's sister is named Jane.
- The user works as an engineer.
- The user owns a bicycle.
- The user dislikes crowded places.

Important rules:

1. Use only information supported by the dialogue.
2. Do not guess.
3. Preserve explicit dates and temporal relationships.
4. Preserve names, quantities, locations, relationships and preferences.
5. Do not automatically generalize one-time behavior into a stable fact.
6. If a durable state changes, retain temporal information needed to
   distinguish earlier and later states.
7. Prefer concise atomic memories.
8. Do not answer any future question.
""".strip()


def joint_typed_prompt(chunk):
    return f"""
Read the entire dialogue chunk and extract EPISODIC and FACT memories
together.

Return valid JSON only:

{{
  "episodic": [
    {{
      "memory": "atomic episodic memory",
      "time": "explicit time if available, otherwise null"
    }}
  ],
  "facts": [
    {{
      "memory": "atomic durable fact"
    }}
  ]
}}

Rules:

- Decide the appropriate memory type while inspecting the dialogue.
- Preserve useful information even if stated only once.
- A particular event belongs in episodic memory.
- A stable preference, relationship, attribute, possession, recurring
  behavior, goal, or durable state belongs in fact memory.
- Avoid unnecessary duplication.
- Do not infer unsupported facts.
- Return JSON only.

DIALOGUE CHUNK
==============

{chunk}
""".strip()


def sequential_typed_prompt(chunk):
    return f"""
Extract long-term memory from this dialogue chunk in TWO DISTINCT
conceptual stages, while producing only ONE final response.

STAGE 1 — EPISODIC MEMORY
=========================

First inspect the ORIGINAL dialogue chunk only for specific:

- events
- actions
- interactions
- experiences
- decisions
- occurrences
- changes over time

Preserve:
- what happened
- who was involved
- when it happened if known
- important context
- chronological changes

Do NOT treat one-time behavior as a durable preference or attribute.

STAGE 2 — FACT MEMORY
=====================

After completing the episodic stage, inspect the ORIGINAL dialogue chunk
again for relatively durable semantic facts:

- stable preferences
- relationships
- occupation
- possessions
- recurring behavior
- persistent goals
- user attributes
- long-term constraints
- durable states

Do NOT merely convert episodic events into facts.

Examples:

"I had sushi yesterday."
-> episodic only

"I usually prefer sushi."
-> fact

"I started working at Company A in June."
-> episodic change event

If the dialogue clearly indicates that Company A is now the user's
durable employment state, it may additionally be preserved as a fact.

FINAL OUTPUT
============

Return valid JSON only:

{{
  "episodic": [
    {{
      "memory": "atomic episodic memory",
      "time": "explicit time if available, otherwise null"
    }}
  ],
  "facts": [
    {{
      "memory": "atomic durable fact"
    }}
  ]
}}

Do not output stage explanations.
Do not output reasoning.
Return JSON only.

DIALOGUE CHUNK
==============

{chunk}
""".strip()


# ============================================================
# JSON PARSING
# ============================================================

def extract_json_object(
    text,
):
    text = text.strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        return json.loads(
            text
        )

    except Exception:
        pass

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start >= 0
        and end > start
    ):

        candidate = text[
            start:end + 1
        ]

        try:
            return json.loads(
                candidate
            )

        except Exception:
            pass

    raise ValueError(
        f"Could not parse JSON:\n"
        f"{text[:1000]}"
    )


def normalize_untyped(
    obj,
):

    memories = []

    for x in (
        obj.get(
            "memories",
            [],
        )
        or []
    ):

        if isinstance(
            x,
            str,
        ):
            text = x.strip()

        elif isinstance(
            x,
            dict,
        ):
            text = str(
                x.get(
                    "memory",
                    "",
                )
            ).strip()

        else:
            continue

        if text:
            memories.append(
                text
            )

    return {
        "memories": memories,
    }


def normalize_typed(
    obj,
):

    episodic = []
    facts = []

    for x in (
        obj.get(
            "episodic",
            [],
        )
        or []
    ):

        if isinstance(
            x,
            str,
        ):
            memory = x.strip()
            timestamp = None

        elif isinstance(
            x,
            dict,
        ):
            memory = str(
                x.get(
                    "memory",
                    "",
                )
            ).strip()

            timestamp = x.get(
                "time"
            )

        else:
            continue

        if memory:
            episodic.append(
                {
                    "memory": memory,
                    "time": timestamp,
                }
            )

    for x in (
        obj.get(
            "facts",
            [],
        )
        or []
    ):

        if isinstance(
            x,
            str,
        ):
            memory = x.strip()

        elif isinstance(
            x,
            dict,
        ):
            memory = str(
                x.get(
                    "memory",
                    "",
                )
            ).strip()

        else:
            continue

        if memory:
            facts.append(
                {
                    "memory": memory,
                }
            )

    return {
        "episodic": episodic,
        "facts": facts,
    }


# ============================================================
# EXTRACTION
# ============================================================

def extract_chunk(
    method,
    chunk,
    model,
    memory_tokens,
):

    if method == "untyped":

        raw = call_llm(
            model=model,
            system=UNTYPED_SYSTEM,
            user=untyped_prompt(
                chunk
            ),
            max_tokens=memory_tokens,
        )

        return normalize_untyped(
            extract_json_object(
                raw
            )
        )

    elif method == "joint_typed":

        raw = call_llm(
            model=model,
            system=TYPED_SYSTEM,
            user=joint_typed_prompt(
                chunk
            ),
            max_tokens=memory_tokens,
        )

        return normalize_typed(
            extract_json_object(
                raw
            )
        )

    elif method == "sequential_typed":

        raw = call_llm(
            model=model,
            system=TYPED_SYSTEM,
            user=sequential_typed_prompt(
                chunk
            ),
            max_tokens=memory_tokens,
        )

        return normalize_typed(
            extract_json_object(
                raw
            )
        )

    else:
        raise ValueError(
            f"Unknown method: "
            f"{method}"
        )


# ============================================================
# FORMAT MEMORIES
# ============================================================

def format_untyped_memories(
    chunk_memories,
):

    lines = []

    for idx, memory in enumerate(
        chunk_memories,
        start=1,
    ):

        lines.append(
            f"## MEMORY CHUNK {idx}"
        )

        memories = memory.get(
            "memories",
            [],
        )

        if not memories:
            lines.append(
                "- None"
            )

        else:

            for m in memories:
                lines.append(
                    f"- {m}"
                )

        lines.append(
            ""
        )

    return "\n".join(
        lines
    )


def format_typed_memories(
    chunk_memories,
):

    lines = []

    for idx, memory in enumerate(
        chunk_memories,
        start=1,
    ):

        lines.append(
            f"## MEMORY CHUNK {idx}"
        )

        lines.append(
            "### EPISODIC"
        )

        episodic = memory.get(
            "episodic",
            [],
        )

        if not episodic:
            lines.append(
                "- None"
            )

        else:

            for e in episodic:

                timestamp = e.get(
                    "time"
                )

                text = e.get(
                    "memory",
                    "",
                )

                if timestamp:
                    lines.append(
                        f"- [{timestamp}] "
                        f"{text}"
                    )
                else:
                    lines.append(
                        f"- {text}"
                    )

        lines.append(
            "### FACT"
        )

        facts = memory.get(
            "facts",
            [],
        )

        if not facts:
            lines.append(
                "- None"
            )

        else:

            for f in facts:

                lines.append(
                    f"- "
                    f"{f.get('memory', '')}"
                )

        lines.append(
            ""
        )

    return "\n".join(
        lines
    )


def format_all_memories(
    method,
    chunk_memories,
):

    if method == "untyped":

        return format_untyped_memories(
            chunk_memories
        )

    return format_typed_memories(
        chunk_memories
    )


# ============================================================
# FINAL QA
# ============================================================

QA_SYSTEM = """
You answer questions using memories extracted from chronological parts of
a dialogue.

The original dialogue is not available.

You must reason across ALL memory chunks before answering.

Important rules:

1. Treat the memory chunks as chronological:
   MEMORY CHUNK 1 occurred before MEMORY CHUNK 2, etc.

2. Pay careful attention to:
   - dates
   - event order
   - changing user states
   - old versus new facts
   - relationships
   - quantities
   - preferences

3. If two memories conflict because something changed over time,
   determine which state is relevant to the question.

4. Do NOT automatically choose the newest fact if the question asks
   about an earlier time.

5. Use only information supported by the supplied memories.

6. Do not explain your reasoning.

7. Return ONLY the shortest direct answer.

8. Do not write phrases such as:
   "The answer is"
   "According to the memory"
   "Based on the information"

Examples:

Question:
What city did I visit in May?

Bad:
According to the memory, you visited Boston in May.

Good:
Boston
""".strip()


def answer_question(
    item,
    method,
    chunk_memories,
    model,
    qa_tokens,
):

    memory_text = format_all_memories(
        method,
        chunk_memories,
    )

    prompt = f"""
MEMORIES
========

{memory_text}

QUESTION DATE
=============

{item.get("question_date", "unknown")}

QUESTION
========

{item["question"]}

Think through the memories internally, then return only the shortest
direct answer.
""".strip()

    return call_llm(
        model=model,
        system=QA_SYSTEM,
        user=prompt,
        max_tokens=qa_tokens,
        temperature=0.0,
    )


# ============================================================
# EM
# ============================================================

def normalize_answer(
    s,
):
    """
    SQuAD-style normalized EM.
    """

    if s is None:
        return ""

    s = str(
        s
    )

    def lower(
        text,
    ):
        return text.lower()

    def remove_punctuation(
        text,
    ):
        exclude = set(
            string.punctuation
        )

        return "".join(
            ch
            for ch in text
            if ch not in exclude
        )

    def remove_articles(
        text,
    ):
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            text,
        )

    def whitespace_fix(
        text,
    ):
        return " ".join(
            text.split()
        )

    return whitespace_fix(
        remove_articles(
            remove_punctuation(
                lower(
                    s
                )
            )
        )
    )


def exact_match(
    prediction,
    gold,
):

    if isinstance(
        gold,
        list,
    ):

        return int(
            any(
                normalize_answer(
                    prediction
                )
                ==
                normalize_answer(
                    x
                )
                for x in gold
            )
        )

    return int(
        normalize_answer(
            prediction
        )
        ==
        normalize_answer(
            gold
        )
    )


# ============================================================
# CACHE
# ============================================================

class JsonCache:

    def __init__(
        self,
        path,
    ):

        self.path = Path(
            path
        )

        self.lock = threading.Lock()

        self.data = {}

        if self.path.exists():

            with open(
                self.path,
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    try:
                        obj = json.loads(
                            line
                        )

                        self.data[
                            obj["cache_key"]
                        ] = obj

                    except Exception:
                        pass

        print(
            f"[cache] loaded "
            f"{len(self.data)} records"
        )

    def get(
        self,
        key,
    ):
        return self.data.get(
            key
        )

    def put(
        self,
        key,
        value,
    ):

        record = {
            "cache_key": key,
            **value,
        }

        with self.lock:

            self.data[
                key
            ] = record

            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with open(
                self.path,
                "a",
                encoding="utf-8",
            ) as f:

                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ============================================================
# SINGLE CONDITION
# ============================================================

def run_condition(
    item,
    method,
    n_chunks,
    memory_model,
    qa_model,
    memory_tokens,
    qa_tokens,
    cache,
):

    qid = item[
        "question_id"
    ]

    cache_key = (
        f"{qid}"
        f"::{method}"
        f"::chunks{n_chunks}"
        f"::{memory_model}"
        f"::{qa_model}"
        f"::mem{memory_tokens}"
        f"::qa{qa_tokens}"
    )

    cached = cache.get(
        cache_key
    )

    if cached is not None:
        return cached

    dialogue_chunks = (
        build_dialogue_chunks(
            item,
            n_chunks,
        )
    )

    # ----------------------------------------
    # MEMORY EXTRACTION
    # exactly ONE call per chunk
    # ----------------------------------------

    chunk_memories = []

    extraction_times = []

    for chunk in dialogue_chunks:

        t0 = time.time()

        memory = extract_chunk(
            method=method,
            chunk=chunk,
            model=memory_model,
            memory_tokens=memory_tokens,
        )

        elapsed = (
            time.time() - t0
        )

        extraction_times.append(
            elapsed
        )

        chunk_memories.append(
            memory
        )

    # ----------------------------------------
    # FINAL REASONING / QA
    # ----------------------------------------

    t1 = time.time()

    prediction = answer_question(
        item=item,
        method=method,
        chunk_memories=chunk_memories,
        model=qa_model,
        qa_tokens=qa_tokens,
    )

    qa_time = (
        time.time() - t1
    )

    gold = item[
        "answer"
    ]

    em = exact_match(
        prediction,
        gold,
    )

    # ----------------------------------------
    # COUNTS
    # ----------------------------------------

    if method == "untyped":

        total_memories = sum(
            len(
                x.get(
                    "memories",
                    [],
                )
            )
            for x in chunk_memories
        )

        num_episodic = None
        num_facts = None

    else:

        num_episodic = sum(
            len(
                x.get(
                    "episodic",
                    [],
                )
            )
            for x in chunk_memories
        )

        num_facts = sum(
            len(
                x.get(
                    "facts",
                    [],
                )
            )
            for x in chunk_memories
        )

        total_memories = (
            num_episodic
            + num_facts
        )

    result = {
        "question_id": qid,
        "question_type": item[
            "question_type"
        ],
        "question": item[
            "question"
        ],
        "gold": gold,
        "prediction": prediction,
        "em": em,

        "method": method,
        "n_chunks": n_chunks,

        "actual_chunks": len(
            dialogue_chunks
        ),

        "total_memories": (
            total_memories
        ),

        "num_episodic": (
            num_episodic
        ),

        "num_facts": (
            num_facts
        ),

        "chunk_memories": (
            chunk_memories
        ),

        "extraction_sec_total": (
            sum(
                extraction_times
            )
        ),

        "extraction_sec_mean": (
            sum(
                extraction_times
            )
            / max(
                1,
                len(
                    extraction_times
                ),
            )
        ),

        "qa_sec": qa_time,
    }

    cache.put(
        cache_key,
        result,
    )

    return result


# ============================================================
# REPORT
# ============================================================

def mean(
    xs,
):

    if not xs:
        return 0.0

    return sum(
        xs
    ) / len(
        xs
    )


def print_main_table(
    results,
    methods,
    chunk_counts,
):

    print()
    print(
        "=" * 120
    )

    print(
        "MAIN RESULTS"
    )

    print(
        "=" * 120
    )

    print(
        f"{'METHOD':22s}"
        f"{'CHUNKS':>8s}"
        f"{'N':>7s}"
        f"{'EM':>10s}"
        f"{'MEMS':>10s}"
        f"{'EPISODIC':>12s}"
        f"{'FACT':>10s}"
        f"{'EXTRACT(s)':>14s}"
        f"{'QA(s)':>10s}"
    )

    print(
        "-" * 120
    )

    for method in methods:

        for chunks in chunk_counts:

            rows = [
                x
                for x in results
                if (
                    x["method"]
                    == method
                    and
                    x["n_chunks"]
                    == chunks
                )
            ]

            if not rows:
                continue

            em = mean(
                [
                    x["em"]
                    for x in rows
                ]
            )

            mems = mean(
                [
                    x["total_memories"]
                    for x in rows
                ]
            )

            extract_time = mean(
                [
                    x[
                        "extraction_sec_total"
                    ]
                    for x in rows
                ]
            )

            qa_time = mean(
                [
                    x["qa_sec"]
                    for x in rows
                ]
            )

            if (
                method
                == "untyped"
            ):
                ep_str = "-"
                fact_str = "-"

            else:

                ep = mean(
                    [
                        x[
                            "num_episodic"
                        ]
                        for x in rows
                    ]
                )

                facts = mean(
                    [
                        x[
                            "num_facts"
                        ]
                        for x in rows
                    ]
                )

                ep_str = (
                    f"{ep:.2f}"
                )

                fact_str = (
                    f"{facts:.2f}"
                )

            print(
                f"{method:22s}"
                f"{chunks:8d}"
                f"{len(rows):7d}"
                f"{em:10.4f}"
                f"{mems:10.2f}"
                f"{ep_str:>12s}"
                f"{fact_str:>10s}"
                f"{extract_time:14.2f}"
                f"{qa_time:10.2f}"
            )

    print(
        "=" * 120
    )


def print_matrix(
    results,
    methods,
    chunk_counts,
):

    print()
    print(
        "=" * 90
    )

    print(
        "EM MATRIX"
    )

    print(
        "=" * 90
    )

    header = (
        f"{'METHOD':28s}"
    )

    for chunks in chunk_counts:
        header += (
            f"{str(chunks)+' chunks':>18s}"
        )

    print(
        header
    )

    print(
        "-" * len(
            header
        )
    )

    for method in methods:

        row = (
            f"{method:28s}"
        )

        for chunks in chunk_counts:

            subset = [
                x
                for x in results
                if (
                    x["method"]
                    == method
                    and
                    x["n_chunks"]
                    == chunks
                )
            ]

            if subset:

                score = mean(
                    [
                        x["em"]
                        for x in subset
                    ]
                )

                row += (
                    f"{score:18.4f}"
                )

            else:
                row += (
                    f"{'N/A':>18s}"
                )

        print(
            row
        )

    print(
        "=" * len(
            header
        )
    )


def print_by_question_type(
    results,
    methods,
    chunk_counts,
):

    qtypes = sorted(
        set(
            x[
                "question_type"
            ]
            for x in results
        )
    )

    print()
    print(
        "=" * 120
    )

    print(
        "BY QUESTION TYPE"
    )

    print(
        "=" * 120
    )

    for chunks in chunk_counts:

        print(
            f"\n### {chunks} CHUNKS"
        )

        header = (
            f"{'QUESTION TYPE':32s}"
        )

        for method in methods:

            header += (
                f"{method:>22s}"
            )

        print(
            header
        )

        print(
            "-" * len(
                header
            )
        )

        for qt in qtypes:

            line = (
                f"{qt:32s}"
            )

            for method in methods:

                subset = [
                    x
                    for x in results
                    if (
                        x[
                            "question_type"
                        ]
                        == qt
                        and
                        x["method"]
                        == method
                        and
                        x["n_chunks"]
                        == chunks
                    )
                ]

                if subset:

                    score = mean(
                        [
                            x["em"]
                            for x in subset
                        ]
                    )

                    line += (
                        f"{score:22.4f}"
                    )

                else:

                    line += (
                        f"{'N/A':>22s}"
                    )

            print(
                line
            )


def print_pairwise(
    results,
    methods,
    chunk_counts,
):

    print()
    print(
        "=" * 110
    )

    print(
        "PAIRWISE WINS"
    )

    print(
        "=" * 110
    )

    for chunks in chunk_counts:

        print(
            f"\n[{chunks} chunks]"
        )

        indexed = defaultdict(
            dict
        )

        for r in results:

            if (
                r["n_chunks"]
                == chunks
            ):
                indexed[
                    r["question_id"]
                ][
                    r["method"]
                ] = r

        for i in range(
            len(
                methods
            )
        ):

            for j in range(
                i + 1,
                len(
                    methods
                ),
            ):

                a = methods[i]
                b = methods[j]

                a_win = 0
                b_win = 0
                tie = 0

                for rows in (
                    indexed.values()
                ):

                    if (
                        a not in rows
                        or b not in rows
                    ):
                        continue

                    ae = rows[a][
                        "em"
                    ]

                    be = rows[b][
                        "em"
                    ]

                    if ae > be:
                        a_win += 1

                    elif be > ae:
                        b_win += 1

                    else:
                        tie += 1

                print(
                    f"{a:22s} "
                    f"vs "
                    f"{b:22s} | "
                    f"{a_win:2d} - "
                    f"{b_win:2d} | "
                    f"tie={tie:2d}"
                )


def print_chunk_comparison(
    results,
    methods,
    chunk_counts,
):

    if len(
        chunk_counts
    ) < 2:
        return

    print()
    print(
        "=" * 110
    )

    print(
        "CHUNK COUNT COMPARISON"
    )

    print(
        "=" * 110
    )

    for method in methods:

        print(
            f"\n[{method}]"
        )

        indexed = defaultdict(
            dict
        )

        for r in results:

            if (
                r["method"]
                == method
            ):

                indexed[
                    r[
                        "question_id"
                    ]
                ][
                    r["n_chunks"]
                ] = r

        for i in range(
            len(
                chunk_counts
            )
        ):

            for j in range(
                i + 1,
                len(
                    chunk_counts
                ),
            ):

                a = chunk_counts[i]
                b = chunk_counts[j]

                a_win = 0
                b_win = 0
                tie = 0

                for rows in (
                    indexed.values()
                ):

                    if (
                        a not in rows
                        or b not in rows
                    ):
                        continue

                    ae = rows[a][
                        "em"
                    ]

                    be = rows[b][
                        "em"
                    ]

                    if ae > be:
                        a_win += 1

                    elif be > ae:
                        b_win += 1

                    else:
                        tie += 1

                print(
                    f"{a} chunks vs "
                    f"{b} chunks | "
                    f"{a} win={a_win:2d} | "
                    f"{b} win={b_win:2d} | "
                    f"tie={tie:2d}"
                )


def print_disagreements(
    results,
    methods,
    chunk_counts,
    max_examples=8,
):

    print()
    print(
        "=" * 120
    )

    print(
        "SAMPLE DISAGREEMENTS"
    )

    print(
        "=" * 120
    )

    indexed = defaultdict(
        dict
    )

    for r in results:

        key = (
            r["question_id"]
        )

        cond = (
            r["method"],
            r["n_chunks"],
        )

        indexed[key][
            cond
        ] = r

    shown = 0

    for qid, rows in (
        indexed.items()
    ):

        scores = [
            r["em"]
            for r in rows.values()
        ]

        if len(
            set(
                scores
            )
        ) <= 1:
            continue

        first = next(
            iter(
                rows.values()
            )
        )

        print()
        print(
            "-" * 120
        )

        print(
            f"QUESTION_ID: {qid}"
        )

        print(
            f"TYPE       : "
            f"{first['question_type']}"
        )

        print(
            f"QUESTION   : "
            f"{first['question']}"
        )

        print(
            f"GOLD       : "
            f"{first['gold']}"
        )

        for chunks in (
            chunk_counts
        ):

            for method in (
                methods
            ):

                key = (
                    method,
                    chunks,
                )

                if key not in rows:
                    continue

                r = rows[
                    key
                ]

                print(
                    f"\n"
                    f"[{method} | "
                    f"{chunks} chunks]"
                )

                print(
                    f"EM   : "
                    f"{r['em']}"
                )

                print(
                    f"PRED : "
                    f"{r['prediction']}"
                )

                print(
                    f"MEMS : "
                    f"{r['total_memories']}"
                )

        shown += 1

        if shown >= max_examples:
            break


# ============================================================
# SAVE
# ============================================================

def save_results(
    path,
    results,
    args,
):

    output = {
        "config": vars(
            args
        ),
        "results": results,
    }

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\n[saved] {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=(
            "google/"
            "gemini-2.5-flash"
        ),
    )

    parser.add_argument(
        "--qa-model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
    )

    parser.add_argument(
        "--chunks",
        nargs="+",
        type=int,
        default=DEFAULT_CHUNKS,
    )

    parser.add_argument(
        "--memory-tokens",
        type=int,
        default=300,
        help=(
            "Maximum output tokens "
            "PER CHUNK extraction."
        ),
    )

    parser.add_argument(
        "--qa-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=(
            "./longmemeval_data"
        ),
    )

    parser.add_argument(
        "--cache",
        type=str,
        default=(
            "./chunked_memory_cache.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "./chunked_memory_results.json"
        ),
    )

    parser.add_argument(
        "--include-abstention",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.qa_model
        is None
    ):
        args.qa_model = (
            args.model
        )

    print(
        "=" * 110
    )

    print(
        "CHUNKED MEMORY ABLATION"
    )

    print(
        "=" * 110
    )

    print(
        f"memory model  : "
        f"{args.model}"
    )

    print(
        f"qa model      : "
        f"{args.qa_model}"
    )

    print(
        f"limit         : "
        f"{args.limit}"
    )

    print(
        f"workers       : "
        f"{args.workers}"
    )

    print(
        f"methods       : "
        f"{args.methods}"
    )

    print(
        f"chunks        : "
        f"{args.chunks}"
    )

    print(
        f"memory tokens : "
        f"{args.memory_tokens}"
        f" per chunk"
    )

    print(
        f"qa tokens     : "
        f"{args.qa_tokens}"
    )

    # ----------------------------------------
    # DATA
    # ----------------------------------------

    dataset_path = (
        download_dataset(
            args.data_dir
        )
    )

    data = load_dataset(
        dataset_path
    )

    subset = balanced_sample(
        data=data,
        limit=args.limit,
        seed=args.seed,
        include_abstention=(
            args.include_abstention
        ),
    )

    # ----------------------------------------
    # CACHE
    # ----------------------------------------

    cache = JsonCache(
        args.cache
    )

    # ----------------------------------------
    # JOBS
    # ----------------------------------------

    jobs = []

    for item in subset:

        for method in (
            args.methods
        ):

            for n_chunks in (
                args.chunks
            ):

                jobs.append(
                    (
                        item,
                        method,
                        n_chunks,
                    )
                )

    print()
    print(
        f"[jobs] conditions="
        f"{len(args.methods) * len(args.chunks)}"
    )

    print(
        f"[jobs] examples="
        f"{len(subset)}"
    )

    print(
        f"[jobs] total="
        f"{len(jobs)}"
    )

    # Estimate extraction calls.
    expected_extract_calls = 0

    for item in subset:

        n_sessions = len(
            item[
                "haystack_sessions"
            ]
        )

        for _method in (
            args.methods
        ):

            for chunks in (
                args.chunks
            ):

                expected_extract_calls += min(
                    chunks,
                    n_sessions,
                )

    expected_qa_calls = len(
        jobs
    )

    print(
        f"[expected extraction calls] "
        f"{expected_extract_calls}"
    )

    print(
        f"[expected QA calls] "
        f"{expected_qa_calls}"
    )

    print(
        f"[expected total calls] "
        f"{expected_extract_calls + expected_qa_calls}"
    )

    # ----------------------------------------
    # RUN
    # ----------------------------------------

    results = []

    with ThreadPoolExecutor(
        max_workers=args.workers,
    ) as executor:

        futures = {}

        for (
            item,
            method,
            n_chunks,
        ) in jobs:

            future = executor.submit(
                run_condition,
                item,
                method,
                n_chunks,
                args.model,
                args.qa_model,
                args.memory_tokens,
                args.qa_tokens,
                cache,
            )

            futures[
                future
            ] = (
                item[
                    "question_id"
                ],
                method,
                n_chunks,
            )

        progress = tqdm(
            as_completed(
                futures
            ),
            total=len(
                futures
            ),
            desc="Running",
        )

        for future in progress:

            (
                qid,
                method,
                n_chunks,
            ) = futures[
                future
            ]

            try:

                result = (
                    future.result()
                )

                results.append(
                    result
                )

                progress.set_postfix(
                    method=method,
                    chunks=n_chunks,
                    em=result["em"],
                )

            except Exception as e:

                print(
                    f"\n[ERROR] "
                    f"qid={qid} "
                    f"method={method} "
                    f"chunks={n_chunks} "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

    # ----------------------------------------
    # SORT
    # ----------------------------------------

    results.sort(
        key=lambda x: (
            str(
                x["question_id"]
            ),
            x["method"],
            x["n_chunks"],
        )
    )

    # ----------------------------------------
    # REPORT
    # ----------------------------------------

    print_main_table(
        results,
        args.methods,
        args.chunks,
    )

    print_matrix(
        results,
        args.methods,
        args.chunks,
    )

    print_by_question_type(
        results,
        args.methods,
        args.chunks,
    )

    print_pairwise(
        results,
        args.methods,
        args.chunks,
    )

    print_chunk_comparison(
        results,
        args.methods,
        args.chunks,
    )

    print_disagreements(
        results,
        args.methods,
        args.chunks,
        max_examples=8,
    )

    # ----------------------------------------
    # SAVE
    # ----------------------------------------

    save_results(
        args.output,
        results,
        args,
    )


if __name__ == "__main__":
    main()
