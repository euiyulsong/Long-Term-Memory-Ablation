#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed LongMemEval chunked-memory ablation.

Main fixes versus the original chunk.py:
1) Disable reasoning for extraction and QA so small max_tokens budgets are not
   consumed by hidden/returned thinking tokens.
2) Use OpenRouter/OpenAI response_format JSON schema for extraction.
3) Treat empty content and invalid JSON as retryable generation failures.
4) Print finish_reason / usage / reasoning length when an empty response occurs.
5) Make sequential_typed actually sequential:
      call #1 -> episodic memories
      call #2 -> fact memories
6) Cache only successful conditions.
7) Record failures instead of silently dropping them from evaluation.
8) Keep SQuAD-style normalized exact match.

Install:
    pip install -U openai huggingface_hub tqdm

Run:
    export OPENROUTER_API_KEY="sk-or-v1-..."

    python3 chunk_fixed.py \
      --limit 50 \
      --workers 16 \
      --model qwen/qwen3.5-35b-a3b \
      --chunks 2 \
      --methods untyped joint_typed sequential_typed \
      --memory-tokens 300 \
      --qa-tokens 16

For debugging empty responses:
    python3 chunk_fixed.py ... --debug-api
"""

import argparse
import json
import os
import random
import re
import string
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download
from openai import OpenAI
from tqdm import tqdm


# =============================================================================
# CONSTANTS
# =============================================================================

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
_print_lock = threading.Lock()


# =============================================================================
# JSON SCHEMAS
# =============================================================================

UNTYPED_JSON_SCHEMA = {
    "name": "untyped_memory",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "memories": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["memories"],
        "additionalProperties": False,
    },
}

TYPED_JSON_SCHEMA = {
    "name": "typed_memory",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "episodic": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "memory": {"type": "string"},
                        "time": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["memory", "time"],
                    "additionalProperties": False,
                },
            },
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "memory": {"type": "string"},
                    },
                    "required": ["memory"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["episodic", "facts"],
        "additionalProperties": False,
    },
}

EPISODIC_JSON_SCHEMA = {
    "name": "episodic_memory",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "episodic": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "memory": {"type": "string"},
                        "time": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "null"},
                            ]
                        },
                    },
                    "required": ["memory", "time"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["episodic"],
        "additionalProperties": False,
    },
}

FACT_JSON_SCHEMA = {
    "name": "fact_memory",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "memory": {"type": "string"},
                    },
                    "required": ["memory"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["facts"],
        "additionalProperties": False,
    },
}


# =============================================================================
# OPENROUTER CLIENT
# =============================================================================

def get_client() -> OpenAI:
    if not hasattr(_thread_local, "client"):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set.\n"
                'export OPENROUTER_API_KEY="sk-or-v1-..."\n'
            )

        _thread_local.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=120.0,
            max_retries=0,  # we handle retries ourselves
        )

    return _thread_local.client


def safe_dump(obj: Any, max_chars: int = 4000) -> str:
    try:
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump()
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = repr(obj)
    return s[:max_chars]


def get_message_text(message: Any) -> str:
    """
    OpenAI-compatible providers normally return message.content as str.
    Be defensive against list-style content.
    """
    content = getattr(message, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue

            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
                continue

            text = getattr(part, "text", None)
            if text:
                parts.append(str(text))

        return "\n".join(parts).strip()

    return ""


def response_debug_info(response: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {}

    try:
        info["id"] = getattr(response, "id", None)
        info["model"] = getattr(response, "model", None)
        info["usage"] = safe_dump(getattr(response, "usage", None), 1500)

        if getattr(response, "choices", None):
            choice = response.choices[0]
            msg = getattr(choice, "message", None)

            info["finish_reason"] = getattr(choice, "finish_reason", None)
            info["content_repr"] = repr(get_message_text(msg))[:1000]

            reasoning = getattr(msg, "reasoning", None)
            if reasoning is None:
                reasoning = getattr(msg, "reasoning_content", None)

            if reasoning is not None:
                reasoning_str = str(reasoning)
                info["reasoning_len"] = len(reasoning_str)
                info["reasoning_preview"] = reasoning_str[:500]
            else:
                info["reasoning_len"] = 0

            info["message"] = safe_dump(msg, 2000)

    except Exception as e:
        info["debug_error"] = repr(e)

    return info


def extract_json_object(text: str) -> Dict[str, Any]:
    if text is None:
        raise ValueError("Could not parse JSON: content=None")

    text = str(text).strip()

    if not text:
        raise ValueError("Could not parse JSON: empty content ''")

    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    ).strip()

    # Exact parse first.
    try:
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise ValueError(
                f"Expected JSON object, got {type(obj).__name__}"
            )
        return obj
    except json.JSONDecodeError:
        pass

    # Then search for first decodable JSON object.
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    raise ValueError(
        "Could not parse JSON. Raw response repr:\n"
        f"{text[:2000]!r}"
    )


def make_response_format(
    json_schema: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if json_schema is None:
        return None

    return {
        "type": "json_schema",
        "json_schema": json_schema,
    }


def call_llm(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
    retries: int = 6,
    json_schema: Optional[Dict[str, Any]] = None,
    debug_api: bool = False,
) -> str:
    """
    Important:
    - reasoning effort is explicitly disabled.
    - empty content is considered failure and retried.
    - malformed JSON is considered failure and retried.
    - structured output is used when json_schema is provided.

    The key bug in the original code was:
        if content is None:
            content = ""
        return content.strip()

    That turns an invalid empty generation into a successful API call.
    """

    client = get_client()
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        response = None

        try:
            request: Dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # OpenRouter unified reasoning parameter.
                # Prevent reasoning tokens from eating the small output budget.
                "extra_body": {
                    "reasoning": {
                        "effort": "none",
                        "exclude": True,
                    }
                },
            }

            response_format = make_response_format(json_schema)
            if response_format is not None:
                request["response_format"] = response_format

            response = client.chat.completions.create(**request)

            if not getattr(response, "choices", None):
                raise ValueError(
                    "LLM returned no choices. "
                    f"response={safe_dump(response)}"
                )

            choice = response.choices[0]
            message = getattr(choice, "message", None)

            if message is None:
                raise ValueError(
                    "LLM returned choice without message. "
                    f"response={safe_dump(response)}"
                )

            content = get_message_text(message)

            if not content:
                info = response_debug_info(response)
                raise ValueError(
                    "LLM returned EMPTY CONTENT. "
                    f"debug={safe_dump(info, 5000)}"
                )

            if json_schema is not None:
                # Validate that it is at least syntactically valid JSON.
                # Schema conformance is additionally enforced by response_format.
                extract_json_object(content)

            if debug_api:
                info = response_debug_info(response)
                with _print_lock:
                    print(
                        "\n[api-ok] "
                        f"model={model} "
                        f"max_tokens={max_tokens} "
                        f"json={json_schema is not None} "
                        f"debug={safe_dump(info, 2500)}"
                    )

            return content

        except Exception as e:
            last_error = e

            info = (
                response_debug_info(response)
                if response is not None
                else {}
            )

            with _print_lock:
                print(
                    f"\n[LLM RETRY {attempt}/{retries}] "
                    f"{type(e).__name__}: {e}"
                )
                if info:
                    print(
                        "[response-debug] "
                        + safe_dump(info, 5000)
                    )

            if attempt >= retries:
                break

            sleep_sec = min(
                0.75 * (2 ** (attempt - 1))
                + random.random() * 0.5,
                12.0,
            )
            time.sleep(sleep_sec)

    raise RuntimeError(
        f"LLM failed after {retries} attempts: "
        f"{type(last_error).__name__ if last_error else 'unknown'}: "
        f"{last_error}"
    )


# =============================================================================
# DATASET
# =============================================================================

def download_dataset(data_dir: str) -> Path:
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = data_dir_path / DATA_FILE

    if path.exists():
        print(f"[dataset] exists: {path}")
        return path

    print("[dataset] downloading...")

    downloaded = hf_hub_download(
        repo_id=DATA_REPO,
        filename=DATA_FILE,
        repo_type="dataset",
        local_dir=str(data_dir_path),
    )

    return Path(downloaded)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    print(f"[dataset] examples={len(data)}")
    return data


def balanced_sample(
    data: List[Dict[str, Any]],
    limit: int,
    seed: int = 42,
    include_abstention: bool = False,
) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in data:
        qid = str(item["question_id"])

        if (
            not include_abstention
            and qid.endswith("_abs")
        ):
            continue

        qtype = item["question_type"]
        groups[qtype].append(item)

    for group in groups.values():
        rnd.shuffle(group)

    qtypes = sorted(groups.keys())

    print("\n[available question types]")
    for qt in qtypes:
        print(f"  {qt:30s}: {len(groups[qt])}")

    selected = []
    idx = 0

    while len(selected) < limit:
        added = False

        for qt in qtypes:
            if idx < len(groups[qt]):
                selected.append(groups[qt][idx])
                added = True

                if len(selected) >= limit:
                    break

        if not added:
            break

        idx += 1

    rnd.shuffle(selected)

    print("\n[selected distribution]")
    counts = Counter(
        x["question_type"]
        for x in selected
    )

    for qt, n in sorted(counts.items()):
        print(f"  {qt:30s}: {n}")

    print(f"  {'TOTAL':30s}: {len(selected)}")

    return selected


# =============================================================================
# SESSION FORMATTING / CHUNKING
# =============================================================================

def format_session(
    session: List[Dict[str, Any]],
    date: Any,
    session_idx: int,
) -> str:
    lines = [
        f"===== SESSION {session_idx} =====",
        f"DATE: {date}",
    ]

    for turn in session:
        role = str(
            turn.get("role", "unknown")
        ).upper()

        content = str(
            turn.get("content", "")
        ).strip()

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(lines)


def split_evenly(
    items: List[Any],
    n_chunks: int,
) -> List[List[Any]]:
    n = len(items)

    if n == 0:
        return []

    n_chunks = max(
        1,
        min(n_chunks, n),
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
        chunks.append(items[start:end])
        start = end

    return chunks


def build_dialogue_chunks(
    item: Dict[str, Any],
    n_chunks: int,
) -> List[str]:
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]

    session_items = []

    for idx, (session, date) in enumerate(
        zip(sessions, dates),
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

    for chunk in split:
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
            "\n\n".join(texts)
        )

    return output


# =============================================================================
# EXTRACTION PROMPTS
# =============================================================================

UNTYPED_SYSTEM = """
You are a deterministic long-term memory extraction engine.

Your task is NOT to answer a question.
Your task is ONLY to extract memories from the supplied dialogue.

Hard requirements:
- Use only information explicitly supported by the dialogue.
- Never guess or invent.
- Preserve names, dates, numbers, quantities, locations, relationships,
  preferences, actions, decisions, and changes over time.
- Preserve temporal information when available.
- If a state changes, keep enough information to distinguish old and new states.
- Prefer concise atomic memories.
- Ignore conversational filler.
- Your response MUST be one valid JSON object matching the requested schema.
- Output no markdown, no code fence, no explanation, and no text before/after JSON.
""".strip()


TYPED_SYSTEM = """
You are a deterministic long-term memory extraction engine.

Your task is NOT to answer a question.
Your task is ONLY to extract memories from the supplied dialogue.

Two memory types exist:

EPISODIC MEMORY:
A specific event, action, experience, interaction, decision, occurrence,
or change situated in time.

FACT MEMORY:
Relatively durable semantic information such as a stable preference,
relationship, occupation, possession, recurring behavior, persistent goal,
attribute, long-term constraint, or durable state.

Hard requirements:
- Use only information explicitly supported by the dialogue.
- Never guess or invent.
- Preserve names, dates, numbers, quantities, locations, relationships,
  preferences, actions, decisions, and changes over time.
- One-time behavior is NOT automatically a durable fact.
- Do not turn every event into a fact.
- Preserve explicit temporal information.
- Prefer concise atomic memories.
- Your response MUST be one valid JSON object matching the requested schema.
- Output no markdown, no code fence, no explanation, and no text before/after JSON.
""".strip()


def untyped_prompt(chunk: str) -> str:
    return f"""
Extract useful long-term memories from the dialogue below.

Return EXACTLY one JSON object with this shape:

{{
  "memories": [
    "atomic memory"
  ]
}}

If there is no useful memory, return:
{{"memories":[]}}

Rules:
1. Each item must be independently understandable.
2. Preserve explicit dates/times when useful.
3. Preserve events and durable facts without categorizing them.
4. Do not infer unsupported details.
5. Do not answer any possible future question.
6. Return JSON only.

DIALOGUE CHUNK
==============
{chunk}
""".strip()


def joint_typed_prompt(chunk: str) -> str:
    return f"""
Read the full dialogue chunk once and extract BOTH memory types jointly.

Return EXACTLY one JSON object with this shape:

{{
  "episodic": [
    {{
      "memory": "atomic episodic memory",
      "time": "explicit date/time if available"
    }}
  ],
  "facts": [
    {{
      "memory": "atomic durable fact"
    }}
  ]
}}

Use null for "time" when no explicit or safely recoverable time is available.

If nothing belongs to a category, return an empty list for that category:
{{
  "episodic": [],
  "facts": []
}}

Rules:
1. Specific events/actions/changes belong in "episodic".
2. Stable preferences/relationships/attributes/possessions/recurring behavior/
   persistent goals/durable states belong in "facts".
3. A memory may appear in both only when the dialogue independently supports
   both an event and a durable resulting state.
4. Avoid unnecessary duplication.
5. Do not infer unsupported information.
6. Return JSON only.

DIALOGUE CHUNK
==============
{chunk}
""".strip()


def episodic_prompt(chunk: str) -> str:
    return f"""
STAGE 1 OF 2: extract EPISODIC memories only.

Return EXACTLY one JSON object with this shape:

{{
  "episodic": [
    {{
      "memory": "atomic episodic memory",
      "time": "explicit date/time if available"
    }}
  ]
}}

Use null for "time" if no explicit or safely recoverable time is available.
If there are no episodic memories, return:
{{"episodic":[]}}

Extract specific:
- events
- actions
- interactions
- experiences
- decisions
- occurrences
- changes over time

Do NOT extract stable semantic facts merely because they can be inferred
from a one-time event.

Return JSON only.

ORIGINAL DIALOGUE CHUNK
=======================
{chunk}
""".strip()


def fact_prompt(chunk: str) -> str:
    return f"""
STAGE 2 OF 2: extract FACT memories only.

IMPORTANT:
Inspect the ORIGINAL dialogue independently.
Do not rely on, rewrite, or summarize Stage 1 output.

Return EXACTLY one JSON object with this shape:

{{
  "facts": [
    {{
      "memory": "atomic durable fact"
    }}
  ]
}}

If there are no durable facts, return:
{{"facts":[]}}

Extract only relatively durable semantic information such as:
- stable preferences
- relationships
- occupation
- possessions
- recurring behavior
- persistent goals
- user attributes
- long-term constraints
- durable states

Do NOT convert one-time events into durable facts.
Do NOT infer unsupported persistence.

Return JSON only.

ORIGINAL DIALOGUE CHUNK
=======================
{chunk}
""".strip()


# =============================================================================
# NORMALIZATION / EXTRACTION
# =============================================================================

def normalize_untyped(
    obj: Dict[str, Any]
) -> Dict[str, List[str]]:
    memories = []

    for x in obj.get("memories", []) or []:
        if isinstance(x, str):
            text = x.strip()
        elif isinstance(x, dict):
            text = str(
                x.get("memory", "")
            ).strip()
        else:
            continue

        if text:
            memories.append(text)

    return {"memories": memories}


def normalize_typed(
    obj: Dict[str, Any]
) -> Dict[str, Any]:
    episodic = []
    facts = []

    for x in obj.get("episodic", []) or []:
        if isinstance(x, str):
            memory = x.strip()
            timestamp = None

        elif isinstance(x, dict):
            memory = str(
                x.get("memory", "")
            ).strip()
            timestamp = x.get("time")

        else:
            continue

        if memory:
            episodic.append(
                {
                    "memory": memory,
                    "time": timestamp,
                }
            )

    for x in obj.get("facts", []) or []:
        if isinstance(x, str):
            memory = x.strip()

        elif isinstance(x, dict):
            memory = str(
                x.get("memory", "")
            ).strip()

        else:
            continue

        if memory:
            facts.append(
                {"memory": memory}
            )

    return {
        "episodic": episodic,
        "facts": facts,
    }


def extract_chunk(
    *,
    method: str,
    chunk: str,
    model: str,
    memory_tokens: int,
    debug_api: bool = False,
) -> Dict[str, Any]:

    if method == "untyped":
        raw = call_llm(
            model=model,
            system=UNTYPED_SYSTEM,
            user=untyped_prompt(chunk),
            max_tokens=memory_tokens,
            json_schema=UNTYPED_JSON_SCHEMA,
            debug_api=debug_api,
        )

        return normalize_untyped(
            extract_json_object(raw)
        )

    if method == "joint_typed":
        raw = call_llm(
            model=model,
            system=TYPED_SYSTEM,
            user=joint_typed_prompt(chunk),
            max_tokens=memory_tokens,
            json_schema=TYPED_JSON_SCHEMA,
            debug_api=debug_api,
        )

        return normalize_typed(
            extract_json_object(raw)
        )

    if method == "sequential_typed":
        # TRUE sequential extraction:
        # two independent model calls over the same original dialogue chunk.

        episodic_raw = call_llm(
            model=model,
            system=TYPED_SYSTEM,
            user=episodic_prompt(chunk),
            max_tokens=memory_tokens,
            json_schema=EPISODIC_JSON_SCHEMA,
            debug_api=debug_api,
        )

        fact_raw = call_llm(
            model=model,
            system=TYPED_SYSTEM,
            user=fact_prompt(chunk),
            max_tokens=memory_tokens,
            json_schema=FACT_JSON_SCHEMA,
            debug_api=debug_api,
        )

        episodic_obj = extract_json_object(
            episodic_raw
        )
        fact_obj = extract_json_object(
            fact_raw
        )

        return normalize_typed(
            {
                "episodic": episodic_obj.get(
                    "episodic",
                    [],
                ),
                "facts": fact_obj.get(
                    "facts",
                    [],
                ),
            }
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


# =============================================================================
# FORMAT MEMORIES
# =============================================================================

def format_untyped_memories(
    chunk_memories: List[Dict[str, Any]]
) -> str:
    blocks = []

    for i, obj in enumerate(
        chunk_memories,
        start=1,
    ):
        lines = [
            f"===== MEMORY CHUNK {i} ====="
        ]

        memories = obj.get(
            "memories",
            [],
        )

        if not memories:
            lines.append("(no extracted memories)")
        else:
            for j, memory in enumerate(
                memories,
                start=1,
            ):
                lines.append(
                    f"- {memory}"
                )

        blocks.append(
            "\n".join(lines)
        )

    return "\n\n".join(blocks)


def format_typed_memories(
    chunk_memories: List[Dict[str, Any]]
) -> str:
    blocks = []

    for i, obj in enumerate(
        chunk_memories,
        start=1,
    ):
        lines = [
            f"===== MEMORY CHUNK {i} =====",
            "[EPISODIC]",
        ]

        episodic = obj.get(
            "episodic",
            [],
        )

        if not episodic:
            lines.append(
                "- (none)"
            )
        else:
            for x in episodic:
                memory = x.get(
                    "memory",
                    "",
                )
                timestamp = x.get(
                    "time"
                )

                if timestamp:
                    lines.append(
                        f"- [{timestamp}] {memory}"
                    )
                else:
                    lines.append(
                        f"- {memory}"
                    )

        lines.append(
            "[FACTS]"
        )

        facts = obj.get(
            "facts",
            [],
        )

        if not facts:
            lines.append(
                "- (none)"
            )
        else:
            for x in facts:
                lines.append(
                    f"- {x.get('memory', '')}"
                )

        blocks.append(
            "\n".join(lines)
        )

    return "\n\n".join(blocks)


def format_all_memories(
    method: str,
    chunk_memories: List[Dict[str, Any]],
) -> str:
    if method == "untyped":
        return format_untyped_memories(
            chunk_memories
        )

    return format_typed_memories(
        chunk_memories
    )


# =============================================================================
# FINAL QA
# =============================================================================

QA_SYSTEM = """
You answer a question using only memories extracted from chronological
parts of a dialogue.

The original dialogue is not available.

Rules:
1. Read all memory chunks before answering.
2. MEMORY CHUNK 1 is earlier than MEMORY CHUNK 2, etc.
3. Pay attention to dates, event order, changing states, relationships,
   quantities, and preferences.
4. If information changed over time, use the state relevant to the question.
5. Do not automatically choose the newest state when the question asks
   about an earlier time.
6. Use only information supported by the memories.
7. Do not explain reasoning.
8. Return ONLY the shortest direct answer.
9. Do not prefix the answer with "The answer is", "According to", etc.
""".strip()


def answer_question(
    *,
    item: Dict[str, Any],
    method: str,
    chunk_memories: List[Dict[str, Any]],
    model: str,
    qa_tokens: int,
    debug_api: bool = False,
) -> str:
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

Return only the shortest direct answer.
""".strip()

    return call_llm(
        model=model,
        system=QA_SYSTEM,
        user=prompt,
        max_tokens=qa_tokens,
        temperature=0.0,
        json_schema=None,
        debug_api=debug_api,
    )


# =============================================================================
# EXACT MATCH
# =============================================================================

def normalize_answer(s: Any) -> str:
    if s is None:
        return ""

    s = str(s)

    def lower(text: str) -> str:
        return text.lower()

    def remove_punctuation(
        text: str
    ) -> str:
        exclude = set(
            string.punctuation
        )
        return "".join(
            ch
            for ch in text
            if ch not in exclude
        )

    def remove_articles(
        text: str
    ) -> str:
        return re.sub(
            r"\b(a|an|the)\b",
            " ",
            text,
        )

    def whitespace_fix(
        text: str
    ) -> str:
        return " ".join(
            text.split()
        )

    return whitespace_fix(
        remove_articles(
            remove_punctuation(
                lower(s)
            )
        )
    )


def exact_match(
    prediction: str,
    gold: Any,
) -> int:
    if isinstance(gold, list):
        return int(
            any(
                normalize_answer(prediction)
                == normalize_answer(x)
                for x in gold
            )
        )

    return int(
        normalize_answer(prediction)
        == normalize_answer(gold)
    )


# =============================================================================
# CACHE
# =============================================================================

class JsonCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.data: Dict[str, Dict[str, Any]] = {}

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
                        record = json.loads(line)
                    except Exception:
                        continue

                    key = record.get(
                        "cache_key"
                    )

                    if key:
                        self.data[key] = record

        print(
            f"[cache] loaded "
            f"{len(self.data)} records"
        )

    def get(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def put(
        self,
        key: str,
        value: Dict[str, Any],
    ) -> None:
        record = {
            "cache_key": key,
            **value,
        }

        with self.lock:
            self.data[key] = record

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


# =============================================================================
# SINGLE CONDITION
# =============================================================================

def extraction_calls_per_chunk(
    method: str
) -> int:
    return (
        2
        if method == "sequential_typed"
        else 1
    )


def run_condition(
    *,
    item: Dict[str, Any],
    method: str,
    n_chunks: int,
    memory_model: str,
    qa_model: str,
    memory_tokens: int,
    qa_tokens: int,
    cache: JsonCache,
    debug_api: bool = False,
) -> Dict[str, Any]:

    qid = str(
        item["question_id"]
    )

    # v2 marker prevents loading incompatible old cache records.
    cache_key = (
        f"v2"
        f"::{qid}"
        f"::{method}"
        f"::chunks{n_chunks}"
        f"::{memory_model}"
        f"::{qa_model}"
        f"::mem{memory_tokens}"
        f"::qa{qa_tokens}"
        f"::reasoning_none"
        f"::structured_json"
    )

    cached = cache.get(
        cache_key
    )

    if (
        cached is not None
        and cached.get("status") == "ok"
    ):
        return cached

    dialogue_chunks = build_dialogue_chunks(
        item,
        n_chunks,
    )

    chunk_memories = []
    extraction_times = []

    for chunk in dialogue_chunks:
        t0 = time.time()

        memory = extract_chunk(
            method=method,
            chunk=chunk,
            model=memory_model,
            memory_tokens=memory_tokens,
            debug_api=debug_api,
        )

        extraction_times.append(
            time.time() - t0
        )

        chunk_memories.append(
            memory
        )

    t1 = time.time()

    prediction = answer_question(
        item=item,
        method=method,
        chunk_memories=chunk_memories,
        model=qa_model,
        qa_tokens=qa_tokens,
        debug_api=debug_api,
    )

    qa_time = (
        time.time() - t1
    )

    gold = item["answer"]

    em = exact_match(
        prediction,
        gold,
    )

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
        "status": "ok",
        "question_id": qid,
        "question_type": item.get(
            "question_type"
        ),
        "question": item.get(
            "question"
        ),
        "question_date": item.get(
            "question_date"
        ),
        "gold": gold,
        "prediction": prediction,
        "em": em,
        "method": method,
        "requested_chunks": n_chunks,
        "actual_chunks": len(
            dialogue_chunks
        ),
        "extraction_calls": (
            len(dialogue_chunks)
            * extraction_calls_per_chunk(
                method
            )
        ),
        "total_memories": total_memories,
        "num_episodic": num_episodic,
        "num_facts": num_facts,
        "extraction_time_sec": sum(
            extraction_times
        ),
        "qa_time_sec": qa_time,
        "total_time_sec": (
            sum(extraction_times)
            + qa_time
        ),
        "chunk_memories": chunk_memories,
    }

    cache.put(
        cache_key,
        result,
    )

    return result


# =============================================================================
# REPORTING
# =============================================================================

def mean(xs: List[float]) -> float:
    if not xs:
        return float("nan")

    return sum(xs) / len(xs)


def summarize_results(
    results: List[Dict[str, Any]]
) -> None:
    print(
        "\n"
        + "=" * 118
    )
    print("RESULT")
    print(
        "=" * 118
    )

    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)

    for r in results:
        grouped[
            (
                r.get("method"),
                r.get("requested_chunks"),
            )
        ].append(r)

    header = (
        f"{'method':24s} "
        f"{'chunks':>6s} "
        f"{'attempt':>8s} "
        f"{'ok':>6s} "
        f"{'fail':>6s} "
        f"{'EM':>8s} "
        f"{'mem':>8s} "
        f"{'calls':>8s} "
        f"{'sec':>9s}"
    )

    print(header)
    print("-" * len(header))

    for (
        method,
        n_chunks,
    ), rows in sorted(
        grouped.items(),
        key=lambda x: (
            str(x[0][0]),
            int(x[0][1]),
        ),
    ):
        oks = [
            x
            for x in rows
            if x.get("status") == "ok"
        ]

        fails = [
            x
            for x in rows
            if x.get("status") != "ok"
        ]

        em = mean(
            [
                float(x["em"])
                for x in oks
            ]
        )

        memories = mean(
            [
                float(
                    x.get(
                        "total_memories",
                        0,
                    )
                )
                for x in oks
            ]
        )

        calls = mean(
            [
                float(
                    x.get(
                        "extraction_calls",
                        0,
                    )
                )
                for x in oks
            ]
        )

        sec = mean(
            [
                float(
                    x.get(
                        "total_time_sec",
                        0,
                    )
                )
                for x in oks
            ]
        )

        print(
            f"{method:24s} "
            f"{n_chunks:6d} "
            f"{len(rows):8d} "
            f"{len(oks):6d} "
            f"{len(fails):6d} "
            f"{em:8.4f} "
            f"{memories:8.1f} "
            f"{calls:8.1f} "
            f"{sec:9.2f}"
        )


def print_error_summary(
    results: List[Dict[str, Any]],
    max_examples: int = 20,
) -> None:
    errors = [
        x
        for x in results
        if x.get("status") != "ok"
    ]

    if not errors:
        print("\n[errors] none")
        return

    print(
        f"\n[errors] {len(errors)}"
    )

    for r in errors[:max_examples]:
        print(
            f"- qid={r.get('question_id')} "
            f"method={r.get('method')} "
            f"chunks={r.get('requested_chunks')} "
            f"{r.get('error_type')}: "
            f"{r.get('error')}"
        )


def save_results(
    path: str,
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    output = {
        "config": vars(args),
        "results": results,
    }

    path_obj = Path(path)
    path_obj.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path_obj,
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
        f"\n[saved] {path_obj}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
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
        default="qwen/qwen3.5-35b-a3b",
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
        help="Maximum visible extraction output tokens per extraction call.",
    )

    parser.add_argument(
        "--qa-tokens",
        type=int,
        default=16,
        help="Maximum visible QA output tokens. Reasoning is disabled.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./longmemeval_data",
    )

    parser.add_argument(
        "--cache",
        type=str,
        default="./chunked_memory_cache_fixed.jsonl",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./chunked_memory_results_fixed.json",
    )

    parser.add_argument(
        "--include-abstention",
        action="store_true",
    )

    parser.add_argument(
        "--debug-api",
        action="store_true",
        help=(
            "Print provider response metadata. "
            "Useful for diagnosing empty content."
        ),
    )

    args = parser.parse_args()

    if args.qa_model is None:
        args.qa_model = args.model

    print(
        "=" * 110
    )
    print(
        "CHUNKED MEMORY ABLATION - FIXED"
    )
    print(
        "=" * 110
    )

    print(
        f"memory model  : {args.model}"
    )
    print(
        f"qa model      : {args.qa_model}"
    )
    print(
        f"limit         : {args.limit}"
    )
    print(
        f"workers       : {args.workers}"
    )
    print(
        f"methods       : {args.methods}"
    )
    print(
        f"chunks        : {args.chunks}"
    )
    print(
        f"memory tokens : {args.memory_tokens} per extraction call"
    )
    print(
        f"qa tokens     : {args.qa_tokens}"
    )
    print(
        "reasoning     : OFF"
    )
    print(
        "structured JSON: ON for extraction"
    )

    dataset_path = download_dataset(
        args.data_dir
    )

    data = load_dataset(
        dataset_path
    )

    subset = balanced_sample(
        data=data,
        limit=args.limit,
        seed=args.seed,
        include_abstention=args.include_abstention,
    )

    cache = JsonCache(
        args.cache
    )

    jobs = []

    for item in subset:
        for method in args.methods:
            for n_chunks in args.chunks:
                jobs.append(
                    (
                        item,
                        method,
                        n_chunks,
                    )
                )

    expected_extract_calls = 0

    for item, method, n_chunks in jobs:
        n_sessions = len(
            item.get(
                "haystack_sessions",
                [],
            )
        )

        actual_chunks = min(
            max(1, n_chunks),
            max(1, n_sessions),
        )

        expected_extract_calls += (
            actual_chunks
            * extraction_calls_per_chunk(
                method
            )
        )

    expected_qa_calls = len(jobs)

    print("\n[jobs]")
    print(
        f"conditions={len(args.methods) * len(args.chunks)}"
    )
    print(
        f"examples={len(subset)}"
    )
    print(
        f"total={len(jobs)}"
    )
    print(
        f"[expected extraction calls] {expected_extract_calls}"
    )
    print(
        f"[expected QA calls] {expected_qa_calls}"
    )
    print(
        f"[expected total calls] "
        f"{expected_extract_calls + expected_qa_calls}"
    )

    results: List[Dict[str, Any]] = []

    def worker(
        job: Any
    ) -> Dict[str, Any]:
        item, method, n_chunks = job

        try:
            return run_condition(
                item=item,
                method=method,
                n_chunks=n_chunks,
                memory_model=args.model,
                qa_model=args.qa_model,
                memory_tokens=args.memory_tokens,
                qa_tokens=args.qa_tokens,
                cache=cache,
                debug_api=args.debug_api,
            )

        except Exception as e:
            qid = str(
                item.get(
                    "question_id",
                    "unknown",
                )
            )

            with _print_lock:
                print(
                    f"\n[ERROR] "
                    f"qid={qid} "
                    f"method={method} "
                    f"chunks={n_chunks} "
                    f"{type(e).__name__}: {e}"
                )

            return {
                "status": "error",
                "question_id": qid,
                "question_type": item.get(
                    "question_type"
                ),
                "question": item.get(
                    "question"
                ),
                "gold": item.get(
                    "answer"
                ),
                "method": method,
                "requested_chunks": n_chunks,
                "error_type": type(e).__name__,
                "error": str(e),
            }

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        future_to_job = {
            executor.submit(
                worker,
                job,
            ): job
            for job in jobs
        }

        with tqdm(
            total=len(jobs),
            desc="Running",
        ) as pbar:

            for future in as_completed(
                future_to_job
            ):
                result = future.result()
                results.append(result)

                oks = sum(
                    1
                    for x in results
                    if x.get("status") == "ok"
                )

                current_em = (
                    sum(
                        x.get("em", 0)
                        for x in results
                        if x.get("status") == "ok"
                    )
                    / max(1, oks)
                )

                pbar.set_postfix(
                    ok=oks,
                    fail=len(results) - oks,
                    em=f"{current_em:.3f}",
                )

                pbar.update(1)

    summarize_results(
        results
    )

    print_error_summary(
        results
    )

    save_results(
        args.output,
        results,
        args,
    )


if __name__ == "__main__":
    main()
