"""Parser and retry-path tests. No API key and no network required.

These are the tests that make the resume claim "a retry for malformed
responses" checkable by anyone who clones the repo.
"""

import json

import pandas as pd
import pytest

from summarise_alerts import build_message, parse_reply, reply_flags, summarise

GOOD = {"summary": "Large transfer to a new beneficiary.",
        "priority": "High", "next_step": "Call the customer to confirm."}

ROW = pd.Series({
    "txn_id": "T00001",
    "customer_id": "C000",
    "amount": 53190.0,
    "channel": "UPI",
    "time": "2026-10-03 14:43:00",
    "rule_fired": "R06",
    "rule_description": "Device change with high amount",
})


class StubClient:
    """Returns scripted replies in order and records how often it was called."""

    def __init__(self, *replies: str):
        self.replies = list(replies)
        self.calls = 0

    def chat(self, system, messages, max_tokens=300):
        self.calls += 1
        if not self.replies:
            raise AssertionError("stub ran out of replies")
        return self.replies.pop(0)


# --- payload -------------------------------------------------------------

def test_message_sends_only_the_four_claimed_fields():
    message = build_message(ROW)
    assert "53,190.00" in message
    assert "UPI" in message
    assert "2026-10-03 14:43:00" in message
    assert "R06" in message
    # identifiers stay local
    assert "T00001" not in message
    assert "C000" not in message


# --- accepting valid replies ---------------------------------------------

def test_plain_json_object():
    assert parse_reply(json.dumps(GOOD))["priority"] == "High"


def test_markdown_fence_is_tolerated():
    text = f"```json\n{json.dumps(GOOD)}\n```"
    assert parse_reply(text)["summary"] == GOOD["summary"]


def test_prose_around_the_object_is_tolerated():
    text = f"Here is the analysis:\n{json.dumps(GOOD)}\nHope that helps!"
    assert parse_reply(text)["next_step"] == GOOD["next_step"]


def test_priority_case_is_normalised():
    data = {**GOOD, "priority": "  hIgH  "}
    assert parse_reply(json.dumps(data))["priority"] == "High"


def test_extra_keys_are_ignored():
    data = {**GOOD, "confidence": 0.9}
    assert parse_reply(json.dumps(data))["priority"] == "High"


# --- rejecting invalid replies -------------------------------------------

@pytest.mark.parametrize("text", [
    "",
    "   ",
    "Sure! Here is my analysis: high priority",
    "{not json}",
    '{"summary": "x", "priority": "High"}',          # missing next_step
    '{"summary": "", "priority": "High", "next_step": "y"}',   # empty value
    json.dumps({**GOOD, "priority": "Critical"}),    # outside the enum
    json.dumps({**GOOD, "priority": "High priority"}),
    '["not", "an", "object"]',
])
def test_malformed_replies_raise(text):
    with pytest.raises(ValueError):
        parse_reply(text)


# --- retry path ----------------------------------------------------------

def test_one_bad_reply_is_retried_then_accepted():
    client = StubClient("Sure! high priority", json.dumps(GOOD))
    record = summarise(client, "sys", ROW)
    assert record["status"] == "ok"
    assert record["attempts"] == 2
    assert record["priority"] == "High"
    assert "no JSON object found" in record["first_error"]
    assert client.calls == 2


def test_persistently_bad_replies_fail_after_three_attempts():
    client = StubClient("nope", "nope", "nope")
    record = summarise(client, "sys", ROW)
    assert record["status"] == "failed"
    assert record["priority"] == "Unrated"
    assert record["attempts"] == 3
    assert client.calls == 3  # MAX_RETRIES=2 -> 3 attempts, no more


def test_retry_feeds_the_error_back_to_the_model():
    client = StubClient("nope", json.dumps(GOOD))
    summarise(client, "sys", ROW)
    # the second call must include the assistant reply and the correction
    assert client.calls == 2


def test_simulate_bad_does_not_spend_an_extra_call():
    client = StubClient(json.dumps(GOOD))
    record = summarise(client, "sys", ROW, simulate_bad=True)
    assert record["attempts"] == 2
    assert record["status"] == "ok"
    assert client.calls == 1  # the bad reply was synthesised, not requested


# --- first-reply shape, used for the V1 vs V2 comparison -----------------

def test_reply_flags_detect_fences_and_prose():
    assert reply_flags("```json\n{}\n```")["first_fenced"] is True
    assert reply_flags(f"Here you go: {json.dumps(GOOD)}")["first_prose"] is True
    assert reply_flags(json.dumps(GOOD)) == {
        "first_fenced": False, "first_prose": False}
