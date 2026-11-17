# GenAI Fraud Alert Summarizer

Takes the alerts produced by a rule-based transaction monitoring engine and asks an LLM to write, for each one, a two-sentence analyst summary, a priority (High / Medium / Low) and a suggested next step — then validates the reply, retries when it is malformed, and ranks the queue into a CSV an analyst can work down.

Companion project: [Transaction Monitoring Rule Engine](https://github.com/Aadhya-dixit/txn-rule-engine) — this repo consumes its `reports/alerts.csv` (279 alerts raised by 8 rules over 10,000 synthetic transactions).

> **Synthetic data.** Every alert comes from the companion repo's seeded generator. No real customer data is used, and the API never receives a customer identifier.

## Headline results

Full run: 40 alerts sampled from 279 (`random_state=42`), prompt V2, `openai/gpt-oss-120b` served by Groq.

| Metric | Result |
|---|---|
| Alerts processed | 40 of 279 |
| Replies parsed and validated | **40/40** |
| Clean on the first attempt | 39 |
| Needed the malformed-response retry | **1** |
| Failed (retried 3 times, still invalid) | **0** |
| Priority split | 16 High · 24 Medium · 0 Low |
| Wall-clock | 121 s at one call every 3 s |

The single retry happened on `T09299`: the first reply was `no JSON object found` (the model emitted no parseable object), the error was fed back, and the second attempt parsed. That is the retry path doing its job on real output, not on a simulated failure.

## How it works

```
data/alerts.csv
      │  sample 40
      ▼
build_message()          amount · channel · time · rule fired
      │                  (no txn_id, no customer_id)
      ▼
system prompt            V1 or V2, both kept in prompts.py
      ▼
LLM API                  Groq / any OpenAI-compatible endpoint
      ▼
parse_reply()  ── invalid ──▶ feed the error back, up to 3 attempts
      │ valid
      ▼
output/checkpoint_v2.jsonl      resume after a crash, never pay twice
      │
      ▼
rank High → Medium → Low, amount descending within a band
      ▼
output/alerts_ranked.csv  +  output/run_summary.json
```

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# .env is gitignored; never commit a key
"GROQ_API_KEY=gsk_..." | Set-Content .env

python -m pytest -q                                  # 20 tests, no API key needed
python summarise_alerts.py --limit 40 --prompt v2    # the full run
python summarise_alerts.py --limit 3 --simulate-bad  # prove the retry path
python evaluate.py                                   # priority vs fraud labels
```

`numpy==2.5.0` is pinned in `requirements.txt` on purpose: 2.5.5's DLL is blocked by a Windows Application Control policy on the machine this was built on.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | key; read from `.env` or the environment |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | point at Ollama (`http://localhost:11434/v1`), OpenRouter, LM Studio |
| `LLM_MODEL` | `openai/gpt-oss-120b` | model ID |
| `LLM_MIN_INTERVAL` | `3.0` | seconds between calls |

That is the provider-agnostic part: the same code runs against a free cloud tier or a local model, so a vendor changing its free catalogue is a configuration change rather than a rewrite. `python llm_client.py --list-models` prints what an endpoint will actually accept.

## The payload

Only four fields leave the machine, exactly as the project claims. Everything else in the CSV — `txn_id` and `customer_id` — stays local.

```
Alert details:
- Amount: Rs.72,309.00
- Channel: Card
- Time: 2026-10-24 16:28:00
- Rule fired: R01 (High amount to new beneficiary)
```

## Prompt design: V1 vs V2

Both prompts are kept in `prompts.py` so the refinement stays reproducible.

**V1** — one sentence:

> Summarise this fraud alert and give a priority (High/Medium/Low) and a next step as JSON.

**V2** — a role, an explicit JSON contract, a closed priority set, a priority rubric, and an instruction not to invent anything the alert does not contain.

Run on the same 10 alerts (`--limit 10`, `random_state=42`):

| | V1 | V2 |
|---|---|---|
| Parsed cleanly on first attempt | 10/10 | 10/10 |
| Retries needed | 0 | 0 |
| Markdown fences / prose around JSON | 0 | 0 |
| **Priority distribution** | **9 High · 1 Medium** | **4 High · 6 Medium** |

The honest finding: **V1 did not fail at formatting.** The format was never the problem — V1 is already good at emitting JSON. V1 failed at *calibration*. It called 9 of 10 alerts High, which makes the ranking useless: if almost everything is High, an analyst has no queue.

V2's rubric is what changed the result. Five of the ten priorities moved, all from High to Medium — for example `T01670`, a Rs.13,257 transfer under rule R03 "odd-hour high amount", which V1 rated High and V2 rated Medium. V2 keeps the genuinely serious patterns at High (`R04` failed-then-success, the largest `R01` and `R06` amounts) while demoting the mid-sized alerts that V1 had swept into the top band.

This is why the guide this project was built from predicted "v1 vs v2 retry counts" and the actual answer is "zero retries either way." Reporting a retry improvement that did not happen would have been fiction. The measurable refinement was the priority distribution.

## Output

`output/alerts_ranked.csv` — 15 columns; the first 6 are the rule engine's own fields, then the three the model produces:

| Column | Meaning |
|---|---|
| `priority` | `High` / `Medium` / `Low`, or `Unrated` if all attempts failed |
| `summary` | ≤2 sentences, model-written |
| `next_step` | one analyst action |
| `attempts` | 1–3 |
| `status` | `ok` / `failed` |
| `first_error` | why the first reply was rejected, empty if it parsed |

The JSON contract the parser enforces:

```json
{
  "summary": "at most 2 sentences",
  "priority": "High | Medium | Low",
  "next_step": "one short action"
}
```

Sample rows, verbatim:

| txn_id | amount | channel | rule | priority | summary |
|---|---|---|---|---|---|
| T07952 | 72,309 | Card | R01 | High | A high-value card transaction of Rs.72,309 was sent to a new beneficiary, triggering rule R01 for high amount to a new payee. |
| T04530 | 48,299 | UPI | R05 | Medium | A UPI transaction of Rs.48,299 triggered a structuring rule, indicating possible attempts to stay below reporting thresholds. |
| T09299 | 29,248 | UPI | R01 | High | A UPI transaction of Rs.29,248 was sent to a previously unused beneficiary, triggering the high-amount new-beneficiary rule. *(attempts = 2)* |

## Do the AI's priorities agree with the fraud labels?

The companion repo labels every transaction `is_fraud`, and all 40 alerts join back on `txn_id`. The labels are joined **after** generation, locally; nothing about them is sent to the API. `evaluate.py` produces:

| AI priority | Alerts | Fraud-labelled | Fraud rate |
|---|---|---|---|
| High | 16 | 4 | **25.0%** |
| Medium | 24 | 12 | **50.0%** |
| All | 40 | 16 | 40.0% |

**The High band contained fewer real frauds than the Medium band.** That is a negative result, and it is reported rather than hidden, because it is the most useful thing in this repo.

The mechanism is visible in the data. The model maps priorities almost perfectly onto *rule identity*:

| Rule fired | Model's priority | Fraud rate in sample |
|---|---|---|
| R01 high amount to new beneficiary | High (8/10) | 20% |
| R06 device change with high amount | High (7/7) | 14% |
| R04 failed attempts then success | High (1/1) | 100% |
| R03 odd-hour high amount | Medium (14/14) | **43%** |
| R08 repeated transfers to same beneficiary | Medium (3/3) | **100%** |
| R05 structuring below reporting limit | Medium (3/3) | 33% |

R01 and R06 *sound* like account takeover, so the model rates them High. R03 fires 14 times in this sample and carries the highest fraud rate of any high-volume rule — but "odd-hour high amount" reads routine, so the model never rates it High. R08 is 3/3 fraud and lands entirely in Medium.

The model cannot know a rule's base rate. It sees four fields and a description, so it reasons from wording. **A priority ranking is not a fraud probability, and using one as if it were would put an analyst's attention in the wrong place.** That is the case for keeping the rule engine's own precision figures alongside the AI's ranking, and for validating any priority scheme against labelled outcomes before it drives a queue.

Two caveats worth stating plainly: n = 40, and the differences (4/16 vs 12/24) are small enough that sampling noise could explain part of the gap. The rule-identity mapping, however, is not noise — it is 40/40 consistent.

## Limitations

- **Synthetic data, and only a sample.** 40 of 279 alerts, from a seeded generator with hand-injected fraud patterns. Real alerts are messier and the class balance is not this friendly.
- **The model only sees four fields,** so it cannot know customer history, usual amount ranges, prior alerts or beneficiary relationships. It reasons from the text of the alert and nothing else.
- **Priorities are model judgments and have not been validated against real analyst decisions.** Against the synthetic fraud labels they were *worse than chance ordering* (25% vs 50% fraud rate, High vs Medium). Treat the ranking as a triage aid, not a risk score.
- **A human analyst makes the final decision.** Nothing here closes an alert, blocks a transaction, or files a report.
- **LLM output varies between runs.** The same alert `T03984` was rated Medium in a one-alert check and High in the V1 run under the same prompt and `temperature=0`. Any number in this README is one run.
- **No Low band appeared.** With only rule-fired alerts as input there is always a stated reason to look, so the model never chose Low in 40 alerts — the three-level scheme collapsed to two in practice.
- **Free-tier limits shape the design.** Groq's free plan allows 8,000 tokens/minute, so the client paces at one call every 3 s. A batch of 279 would take about 14 minutes, which is why the sample is 40.
- **Prompt-injection surface.** `rule_description` is interpolated into the prompt. It comes from a trusted config file here, but a free-text alert field would be an injection vector.

## Design decisions

**Why JSON rather than free text?** Free text cannot be sorted. The priority has to be a value the CSV can sort and count, not a phrase a human has to interpret. A closed set of three labels also makes the output comparable across 40 rows.

**What happens when the model does not follow the format?** `parse_reply` strips markdown fences, locates the outermost object, requires all three keys with non-empty values, and rejects anything outside the priority enum. On rejection the model's own reply is sent back with the parse error and a one-line instruction, up to two more attempts. The failure is recorded in `first_error` rather than being silently repaired, so the retry rate is measurable. After three attempts the row is written as `Unrated` / `failed` and the run continues — one bad alert does not lose the other 39.

**Why two separate retry mechanisms?** A 429 is not the same event as a malformed reply. Rate limits are retried with exponential backoff, honouring `Retry-After`, and never count as a parse retry; authentication and model-not-found errors are not retried at all, because they fail identically every time and retrying only hides the real message.

**Why pace at 3 seconds?** Free tiers are token-per-minute limited, not just request-per-minute limited. 8,000 tokens/minute divided by ~350 tokens per alert is about 22 calls a minute, so the client paces just under that. Without pacing the run produces 429s that look like a broken parser.

**Why send no customer identifier?** The four fields are enough to write the summary, and the free tier may use prompts to improve the provider's products. Leaving `customer_id` and `txn_id` behind keeps the request to what an analyst sees on the alert and removes the obvious privacy objection.

## Reproducing these figures

The sample is deterministic (`random_state=42`), but the model is not (see the variance note above), so a re-run will produce the same 40 alerts and slightly different text. `output/run_summary.json` is written by the run itself and is the only source for the numbers in this README — nothing here is typed in by hand. Tests run without a key: `python -m pytest -q`.
