"""Send rule-engine fraud alerts to an LLM and rank the replies.

Input : data/alerts.csv  (txn_id, amount, channel, time, rule_fired, ...)
Output: output/alerts_ranked.csv  (High -> Medium -> Low, amount desc within band)
        output/run_summary.json   (the only numbers the README quotes)

Only four fields reach the API -- amount, channel, time and rule fired. The
customer_id and txn_id stay local, so the prompt contains no direct identifier
and every summary is grounded in what an analyst can see on the alert itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

import prompts
from llm_client import ChatClient, LLMError

PRIORITIES = {"High": 0, "Medium": 1, "Low": 2}
UNRATED = "Unrated"
MAX_RETRIES = 2  # -> at most 3 attempts per alert

# Four fields, no identifiers. Keep this the single place the payload is built.
MESSAGE_TEMPLATE = (
    "Alert details:\n"
    "- Amount: Rs.{amount:,.2f}\n"
    "- Channel: {channel}\n"
    "- Time: {time}\n"
    "- Rule fired: {rule_fired} ({rule_description})"
)


def build_message(row: pd.Series) -> str:
    """The exact user message sent to the model. Nothing else is transmitted."""
    return MESSAGE_TEMPLATE.format(
        amount=float(row["amount"]),
        channel=row["channel"],
        time=row["time"],
        rule_fired=row["rule_fired"],
        rule_description=row["rule_description"],
    )


def parse_reply(text: str) -> dict:
    """Validate one model reply, or raise ValueError explaining what was wrong.

    Tolerates markdown fences and chatter around the object, because a human
    reading the reply would tolerate them too. Everything else is rejected:
    no object, malformed JSON, a missing key, an empty value, or a priority
    outside the closed set.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty reply")

    cleaned = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON was not an object")

    for key in ("summary", "priority", "next_step"):
        if key not in data:
            raise ValueError(f"missing key: {key}")
        if not isinstance(data[key], str) or not data[key].strip():
            raise ValueError(f"empty key: {key}")

    data["summary"] = data["summary"].strip()
    data["next_step"] = data["next_step"].strip()
    data["priority"] = data["priority"].strip().capitalize()
    if data["priority"] not in PRIORITIES:
        raise ValueError(f"invalid priority: {data['priority']}")

    return {
        "summary": data["summary"],
        "priority": data["priority"],
        "next_step": data["next_step"],
    }


def reply_flags(text: str) -> dict:
    """Shape of the *first* reply, so V1 and V2 can be compared on evidence."""
    raw = text or ""
    brace = raw.find("{")
    before = re.sub(r"```(?:json)?", "", raw[:brace] if brace > 0 else "").strip()
    return {
        "first_fenced": "```" in raw,
        "first_prose": bool(before),
    }


def summarise(
    client,
    system: str,
    row: pd.Series,
    max_retries: int = MAX_RETRIES,
    simulate_bad: bool = False,
) -> dict:
    """One alert in, one validated record out.

    Malformed replies are retried with the parse error fed back to the model,
    which is the failure mode the JSON contract is meant to prevent.
    """
    messages = [{"role": "user", "content": build_message(row)}]
    flags: dict | None = None
    first_error = ""

    for attempt in range(max_retries + 1):
        if simulate_bad and attempt == 0:
            # Short-circuit before the API call, so exercising the retry path
            # costs nothing and cannot be mistaken for a real failure.
            text = "Sure! Here is my analysis: high priority"
        else:
            text = client.chat(system=system, messages=messages, max_tokens=300)

        if attempt == 0:
            flags = reply_flags(text)

        try:
            parsed = parse_reply(text)
        except ValueError as exc:
            if attempt == 0:
                first_error = str(exc)
            if attempt == max_retries:
                break
            messages += [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"That reply was invalid ({exc}). "
                        "Reply again with ONLY the JSON object."
                    ),
                },
            ]
            continue

        return {
            **parsed,
            "attempts": attempt + 1,
            "status": "ok",
            "first_error": first_error,
            **(flags or {}),
        }

    return {
        "summary": "",
        "priority": UNRATED,
        "next_step": "",
        "attempts": max_retries + 1,
        "status": "failed",
        "first_error": first_error or "invalid reply",
        **(flags or {}),
    }


def load_checkpoint(path: Path) -> dict[str, dict]:
    """Completed alerts from an earlier run, keyed by txn_id."""
    if not path.exists():
        return {}
    done: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a torn final line from an interrupted run
        done[record["txn_id"]] = record
    return done


def append_checkpoint(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def select_alerts(path: Path, limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    alerts = pd.read_csv(path)
    chosen = alerts.sample(min(limit, len(alerts)), random_state=42)
    return alerts, chosen.reset_index(drop=True)


def write_summary(
    path: Path,
    *,
    alerts_processed: int,
    alerts_available: int,
    version: str,
    client,
    frame: pd.DataFrame,
    elapsed: float,
) -> dict:
    attempts = frame["attempts"]
    first_invalid = int((frame["first_error"].fillna("") != "").sum())
    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_version": version.upper(),
        "model": client.model,
        "endpoint_host": client.endpoint_host,
        "alerts_available": alerts_available,
        "alerts_processed": alerts_processed,
        "ok": int((frame["status"] == "ok").sum()),
        "failed": int((frame["status"] == "failed").sum()),
        "needed_retry": int((attempts > 1).sum()),
        "max_attempts": int(attempts.max()) if len(attempts) else 0,
        "first_reply_invalid": first_invalid,
        "first_reply_fenced": int(frame["first_fenced"].fillna(False).sum()),
        "first_reply_prose": int(frame["first_prose"].fillna(False).sum()),
        "priority_counts": {
            str(k): int(v) for k, v in frame["priority"].value_counts().items()
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", default="data/alerts.csv", type=Path)
    parser.add_argument("--outdir", default="output", type=Path)
    parser.add_argument("--limit", default=40, type=int,
                        help="alerts to sample (40 by default, 0 for all)")
    parser.add_argument("--prompt", default="v2", choices=sorted(prompts.PROMPTS))
    parser.add_argument("--max-retries", default=MAX_RETRIES, type=int)
    parser.add_argument("--simulate-bad", action="store_true",
                        help="force one malformed first reply to exercise the retry path")
    parser.add_argument("--fresh", action="store_true",
                        help="ignore the checkpoint and re-call the API for every alert")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    system = prompts.PROMPTS[args.prompt]
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    if not os.getenv("GROQ_API_KEY") and not os.getenv("LLM_API_KEY") \
            and not Path(".env").exists():
        print("error: no API key. Put GROQ_API_KEY=... in .env", file=sys.stderr)
        return 2

    try:
        client = ChatClient()
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    limit = args.limit if args.limit > 0 else 10 ** 9
    alerts, chosen = select_alerts(args.input, limit)

    checkpoint = args.outdir / f"checkpoint_{args.prompt}.jsonl"
    done = {} if args.fresh or args.simulate_bad else load_checkpoint(checkpoint)

    suffix = "_simulated" if args.simulate_bad else ""
    out_csv = args.outdir / f"alerts_ranked{suffix}.csv"
    out_json = args.outdir / f"run_summary{suffix}.json"

    print(f"model      : {client.model} via {client.endpoint_host}")
    print(f"prompt     : {args.prompt.upper()}")
    print(f"alerts     : {len(chosen)} of {len(alerts)} "
          f"(sampled with random_state=42)")
    print(f"pacing     : {client.min_interval:g}s between calls")
    if done:
        print(f"resuming   : {len(done)} alert(s) already in {checkpoint.name}")
    print()

    started = time.monotonic()
    results: list[dict] = []
    for index, row in chosen.iterrows():
        txn_id = row["txn_id"]
        cached = done.get(txn_id)
        if cached and cached.get("model") == client.model:
            results.append({k: cached[k] for k in (
                "summary", "priority", "next_step", "attempts", "status",
                "first_error", "first_fenced", "first_prose")})
            print(f"{index + 1:>3}/{len(chosen)}  {cached['priority']:<7} "
                  f"cached    {txn_id}")
            continue

        try:
            record = summarise(
                client, system, row,
                max_retries=args.max_retries,
                simulate_bad=args.simulate_bad,
            )
        except LLMError as exc:
            print(f"\nerror: {exc}", file=sys.stderr)
            print("Progress is checkpointed; re-run to continue.", file=sys.stderr)
            return 1

        results.append(record)
        if not args.simulate_bad:
            append_checkpoint(checkpoint, {
                "txn_id": txn_id,
                "prompt": args.prompt,
                "model": client.model,
                **record,
            })

        note = f"retry ({record['first_error']})" if record["attempts"] > 1 else ""
        print(f"{index + 1:>3}/{len(chosen)}  {record['priority']:<7} "
              f"attempts={record['attempts']}  {txn_id}  {note}".rstrip())

    elapsed = time.monotonic() - started

    block = pd.DataFrame(results)[
        ["priority", "summary", "next_step", "attempts", "status",
         "first_error", "first_fenced", "first_prose"]
    ]
    out = pd.concat([chosen.reset_index(drop=True), block], axis=1)

    rank = out["priority"].map({**PRIORITIES, UNRATED: len(PRIORITIES)})
    out = (
        out.assign(_rank=rank)
        .sort_values(["_rank", "amount"], ascending=[True, False])
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )

    columns = [
        "txn_id", "amount", "channel", "time", "rule_fired", "rule_description",
        "priority", "summary", "next_step", "attempts", "status",
        "first_error", "first_fenced", "first_prose", "customer_id",
    ]
    out = out[[c for c in columns if c in out.columns]]
    out.to_csv(out_csv, index=False)

    summary = write_summary(
        out_json,
        alerts_processed=len(out),
        alerts_available=len(alerts),
        version=args.prompt,
        client=client,
        frame=out,
        elapsed=elapsed,
    )

    print()
    print(f"priority   : {summary['priority_counts']}")
    print(f"parsed ok  : {summary['ok']}/{summary['alerts_processed']}"
          f"  (first attempt clean: "
          f"{summary['alerts_processed'] - summary['first_reply_invalid']})")
    print(f"retries    : {summary['needed_retry']}   failed: {summary['failed']}")
    print(f"elapsed    : {summary['elapsed_seconds']}s")
    print(f"written    : {out_csv}")
    print(f"summary    : {out_json}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
