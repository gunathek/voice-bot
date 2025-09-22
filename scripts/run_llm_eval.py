#!/usr/bin/env python3
"""
Run batch LLM calls for manual evaluation against expected answers.

Reads Q/A pairs from a CSV or JSON file and prints, for each item,
the question, expected answer, and the model's output.

Defaults integrate with this repo's current setup:
- Loads `.env.local` so SARVAM_* env vars are picked up
- Uses the same system prompt file at `prompts/system_prompt.md`
- Calls Sarvam's OpenAI-compatible Chat Completions API

Usage examples:
  python scripts/run_llm_eval.py --input eval/questions.csv
  python scripts/run_llm_eval.py --input eval/questions.json --model sarvam-m
  python scripts/run_llm_eval.py --input eval/questions.csv --output eval/out.jsonl
  # Rate limit to 30 requests/min, with retries on 429
  python scripts/run_llm_eval.py --input eval/questions.json --rpm 30 --retry-max 5 --verbose

Input format:
- CSV: columns `question`, `expected` (optional `id`)
- JSON: list of objects with keys `question`, `expected` (optional `id`)

Environment variables (loaded from .env.local if present):
- SARVAM_LLM_API_KEY or SARVAM_API_KEY
- SARVAM_LLM_BASE_URL (default: https://api.sarvam.ai/v1)
- SARVAM_LLM_MODEL (default: sarvam-m)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import random
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv


DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "system_prompt.md"


@dataclass
class QAItem:
    id: str
    question: str
    expected: str


class APIError(RuntimeError):
    """Raised for non-200 API responses with access to status and headers."""

    def __init__(self, status_code: int, message: str, headers: Optional[Dict[str, str]] = None):
        super().__init__(f"API {status_code}: {message}")
        self.status_code = status_code
        self.headers = dict(headers or {})


def _load_env() -> None:
    # Load .env.local if present to match agent.py behavior
    load_dotenv(".env.local")


def _read_system_prompt(path: Optional[str]) -> str:
    p = Path(path) if path else DEFAULT_SYSTEM_PROMPT_PATH
    try:
        return p.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise SystemExit(f"System prompt file not found: {p}")


def _read_input_rows(path: str) -> List[QAItem]:
    in_path = Path(path)
    if not in_path.exists():
        raise SystemExit(f"Input file not found: {in_path}")

    items: List[QAItem] = []
    if in_path.suffix.lower() == ".csv":
        with in_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"question", "expected"}
            if not required.issubset(set((reader.fieldnames or []))):
                raise SystemExit(
                    f"CSV must contain columns: {sorted(required)}. Got: {reader.fieldnames}"
                )
            for i, row in enumerate(reader, start=1):
                q = (row.get("question") or "").strip()
                e = (row.get("expected") or "").strip()
                if not q:
                    continue
                items.append(
                    QAItem(
                        id=(row.get("id") or str(i)).strip(),
                        question=q,
                        expected=e,
                    )
                )
    elif in_path.suffix.lower() in {".json", ".jsonl"}:
        with in_path.open("r", encoding="utf-8") as f:
            if in_path.suffix.lower() == ".jsonl":
                data = [json.loads(line) for line in f if line.strip()]
            else:
                data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit("JSON input must be a list of objects")
        for i, obj in enumerate(data, start=1):
            if not isinstance(obj, dict):
                continue
            q = (obj.get("question") or "").strip()
            e = (obj.get("expected") or "").strip()
            if not q:
                continue
            items.append(
                QAItem(
                    id=str(obj.get("id") or i),
                    question=q,
                    expected=e,
                )
            )
    else:
        raise SystemExit("Unsupported input format. Use .csv, .json, or .jsonl")

    if not items:
        raise SystemExit("No valid rows found in input file")
    return items


def _build_headers(api_key: str) -> Dict[str, str]:
    # Sarvam's OpenAI-compatible API typically uses Bearer auth
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _chat_completions(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_content: str,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    timeout: float = 60.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    resp = requests.post(url, headers=_build_headers(api_key), json=payload, timeout=timeout)
    if resp.status_code != 200:
        # surface structured error if present
        try:
            err = resp.json()
            msg = err.get("error") or err
        except Exception:
            msg = resp.text
        raise APIError(resp.status_code, str(msg), headers=resp.headers)
    data = resp.json()

    # OpenAI-style response
    try:
        content = data["choices"][0]["message"]["content"]
        return str(content or "").strip()
    except Exception:
        # Fallback for any variant
        return json.dumps(data)


def _ensure_out_dir(path: Optional[str]) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    if p.suffix:
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)
    return p


def _parse_retry_after(headers: Dict[str, str]) -> Optional[float]:
    """Return seconds to wait from Retry-After header, if present/parsable."""
    if not headers:
        return None
    for key in ("Retry-After", "retry-after", "RETRY-AFTER"):
        if key in headers:
            val = headers.get(key)
            if not val:
                continue
            val = str(val).strip()
            # Seconds format
            if val.isdigit():
                try:
                    return max(0.0, float(int(val)))
                except Exception:
                    pass
            # HTTP-date format
            try:
                dt = parsedate_to_datetime(val)
                if dt is not None:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    now = datetime.now(timezone.utc)
                    return max(0.0, (dt - now).total_seconds())
            except Exception:
                pass
    return None


def _call_with_retries(fn, *, max_retries: int, backoff: float, backoff_max: float,
                       retry_statuses: Iterable[int], respect_retry_after: bool, verbose: bool) -> str:
    attempt = 0
    while True:
        try:
            return fn()
        except APIError as e:
            attempt += 1
            if e.status_code not in retry_statuses or attempt > max_retries:
                raise
            sleep_s: Optional[float] = None
            if respect_retry_after:
                sleep_s = _parse_retry_after(e.headers)
            if sleep_s is None:
                sleep_s = min(backoff_max, backoff * (2 ** (attempt - 1)))
            # Add jitter up to 25% to reduce thundering herd
            jitter = random.uniform(0, 0.25 * sleep_s)
            if verbose:
                print(f"Retryable API error {e.status_code}. Sleeping {sleep_s + jitter:.2f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s + jitter)
        except (requests.Timeout, requests.ConnectionError) as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = min(backoff_max, backoff * (2 ** (attempt - 1)))
            jitter = random.uniform(0, 0.25 * sleep_s)
            if verbose:
                print(f"Network error: {e}. Sleeping {sleep_s + jitter:.2f}s (attempt {attempt}/{max_retries})")
            time.sleep(sleep_s + jitter)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run LLM eval calls against expected answers")
    parser.add_argument("--input", required=True, help="Path to CSV/JSON(.json/.jsonl) with question/expected pairs")
    parser.add_argument("--output", help="Optional output JSONL file to save results")
    parser.add_argument("--system-prompt", dest="system_prompt_path", help="Path to system prompt file (default: prompts/system_prompt.md)")
    parser.add_argument("--system-prompt-text", help="Override system prompt content as a literal string")
    parser.add_argument("--model", help="Model name (default from SARVAM_LLM_MODEL or sarvam-m)")
    parser.add_argument("--base-url", help="Base URL (default from SARVAM_LLM_BASE_URL or https://api.sarvam.ai/v1)")
    parser.add_argument("--api-key", help="API key (default from SARVAM_LLM_API_KEY or SARVAM_API_KEY)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for output")
    parser.add_argument("--sleep", type=float, default=0.0, help="Fixed seconds to sleep between requests (overridden by --rps/--rpm)")
    parser.add_argument("--rps", type=float, default=None, help="Rate limit in requests per second (mutually combinable with --rpm; slowest wins)")
    parser.add_argument("--rpm", type=float, default=None, help="Rate limit in requests per minute (mutually combinable with --rps; slowest wins)")
    parser.add_argument("--retry-max", type=int, default=4, help="Max retry attempts for retryable errors (default: 4)")
    parser.add_argument("--retry-backoff", type=float, default=1.5, help="Initial backoff seconds for retries (default: 1.5)")
    parser.add_argument("--retry-backoff-max", type=float, default=60.0, help="Max backoff seconds for retries (default: 60)")
    parser.add_argument("--retry-statuses", type=str, default="429,502,503,504,408", help="Comma-separated HTTP statuses to retry (default: 429,502,503,504,408)")
    parser.add_argument("--no-retry-after", action="store_true", help="Ignore Retry-After header on retryable responses")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to run")
    parser.add_argument("--timeout", type=float, default=60.0, help="Request timeout in seconds")
    parser.add_argument("--verbose", action="store_true", help="Print extra request/response info")

    args = parser.parse_args(argv)

    _load_env()

    base_url = args.base_url or os.getenv("SARVAM_LLM_BASE_URL", "https://api.sarvam.ai/v1")
    api_key = args.api_key or os.getenv("SARVAM_LLM_API_KEY") or os.getenv("SARVAM_API_KEY")
    model = args.model or os.getenv("SARVAM_LLM_MODEL", "sarvam-m")
    if not api_key:
        print("Missing API key. Set SARVAM_LLM_API_KEY or SARVAM_API_KEY, or pass --api-key.", file=sys.stderr)
        return 2

    system_prompt = args.system_prompt_text if args.system_prompt_text else _read_system_prompt(args.system_prompt_path)

    items = _read_input_rows(args.input)
    if args.limit is not None:
        items = items[: max(0, args.limit)]

    out_path: Optional[Path] = None
    out_file = None
    if args.output:
        out_path = _ensure_out_dir(args.output)
        if out_path.is_dir():
            ts = time.strftime("%Y%m%d-%H%M%S")
            out_path = out_path / f"run-{ts}.jsonl"
        out_file = out_path.open("w", encoding="utf-8")

    # Compute throttling schedule
    min_interval: Optional[float] = None
    if args.rps is not None and args.rps > 0:
        min_interval = 1.0 / float(args.rps)
    if args.rpm is not None and args.rpm > 0:
        rpm_interval = 60.0 / float(args.rpm)
        if min_interval is None:
            min_interval = rpm_interval
        else:
            # pick the slowest (largest interval)
            min_interval = max(min_interval, rpm_interval)
    next_allowed_ts = time.monotonic()

    # Parse retry statuses
    retry_statuses = set()
    for s in (args.retry_statuses or "").split(","):
        s = s.strip()
        if not s:
            continue
        try:
            retry_statuses.add(int(s))
        except ValueError:
            pass

    try:
        for idx, item in enumerate(items, start=1):
            try:
                # Throttle if needed
                if min_interval is not None:
                    now = time.monotonic()
                    if now < next_allowed_ts:
                        sleep_for = next_allowed_ts - now
                        if args.verbose:
                            print(f"Rate limit: sleeping {sleep_for:.3f}s before next request")
                        time.sleep(sleep_for)
                    next_allowed_ts = max(now, next_allowed_ts) + min_interval

                # Call with retries
                def _do_call():
                    return _chat_completions(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        system_prompt=system_prompt,
                        user_content=item.question,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                    )

                model_out = _call_with_retries(
                    _do_call,
                    max_retries=max(0, int(args.retry_max)),
                    backoff=max(0.1, float(args.retry_backoff)),
                    backoff_max=max(0.1, float(args.retry_backoff_max)),
                    retry_statuses=retry_statuses or {429, 502, 503, 504, 408},
                    respect_retry_after=not args.no_retry_after,
                    verbose=bool(args.verbose),
                )
            except Exception as e:
                model_out = f"<ERROR: {e}>"

            print("-" * 80)
            print(f"ID: {item.id}")
            print(f"Q:  {item.question}")
            print(f"Expected: {item.expected}")
            print(f"Model:    {model_out}")

            if out_file is not None:
                rec = {
                    "id": item.id,
                    "question": item.question,
                    "expected": item.expected,
                    "model": model_out,
                    "model_name": model,
                    "base_url": base_url,
                    "temperature": args.temperature,
                    "max_tokens": args.max_tokens,
                    "system_prompt_path": args.system_prompt_path or str(DEFAULT_SYSTEM_PROMPT_PATH),
                }
                out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if args.verbose:
                sys.stdout.flush()

            # Backward-compatible fixed sleep only when no rate limit is set
            if min_interval is None and args.sleep > 0 and idx < len(items):
                time.sleep(args.sleep)
    finally:
        if out_file is not None:
            out_file.close()
            print("-" * 80)
            print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
