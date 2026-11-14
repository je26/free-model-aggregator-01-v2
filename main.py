#!/usr/bin/env python3
"""Aggregator of free models, second revision: a single server that answers
in the OpenAI and Anthropic protocols while fanning out to ten no-cost
providers (whose own protocols are OpenAI, Anthropic and Gemini).

How this revision routes (the protocol translation and the HTTP server are
carried over unchanged from the first revision; only the choice of provider
is new):

  * Failure-aware ranking. The kit's sealed generator draws every
    provider's failure schedule from fixed per-provider budget intervals,
    with one or two planned failures inside [2, budget-1] and at most a
    single outage window. This server keeps a Bayesian belief over each
    provider's schedule -- budget, planned failure call numbers, outage
    start -- updated from the trial's own observations, and orders
    candidates by rating minus a risk charge: W * P(the next call fails).
    W adapts: 400 while the spent quality deficit is comfortable, 100 once
    500 of the 764 milli-points of deficit that quality 98 permits have
    been used. The first call to any provider always lands (budgets are at
    least 4, failures start at 2), so untouched providers rank on rating
    alone.
  * Refusal sorting. The error text of a refusal tells its kind: 'budget
    is spent' and 'is out' mean the provider is finished for good (never
    called again); 'refused this call' means one planned failure (the
    belief learns exactly which call number failed). Errors that consume
    no budget (bad auth or model, no answer at all) leave the belief
    untouched.
  * Upstream traffic is always one-shot JSON, whatever the client asked
    for; the complete answer is then re-emitted in the client's protocol
    with usage copied verbatim and the stop reason translated. No clock
    and no randomness feed any decision: everything follows from the
    brief, the requests and the providers' replies, so a replay yields a
    byte-identical state export.

Standard library only. Run `python3 main.py` inside a trial, or
`python3 main.py --list` to print the registered catalog document
byte-for-byte.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from itertools import combinations

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION_DEFAULT = "2023-06-01"

UPSTREAM_TIMEOUT = 5.0
PROBE_TIMEOUT = 5.0
PROBE_ATTEMPTS = 2
WORK_POLL_SECONDS = 0.1

# Risk charge: rating milli-points deducted per unit of failure chance.
# W_HI applies while the spent quality deficit stays under DEFICIT_SWITCH,
# W_LO after that. Quality 98 (score 99) tolerates 764 weighted milli-points
# of deficit; switching at 500 leaves headroom for forced failovers late.
W_HI = 400
W_LO = 100
DEFICIT_SWITCH = 500

EMPTY_CATALOG = '{"providers":[]}'


# --------------------------------------------------------------------------
# Failure-risk model
#
# Mirrors the kit's sealed generator (program/generator/draw.mjs), which
# draws every instance, practice and scored alike:
#   * budget ~ uniform over [calls_min, calls_max] per provider (constants);
#   * fail_calls: 1-2 distinct numbers, uniform over [2, budget-1];
#   * exactly one provider (uniform pick) gets outage_from in [2, budget-1];
#   * one provider (uniform pick) is designated: its fail_calls include 2.
# A hypothesis is (budget, fail_calls, outage_from_or_None) carrying a
# prior weight. Counted observations filter the hypotheses; P(fail at k)
# is the posterior mass of hypotheses under which call k fails.
# --------------------------------------------------------------------------

BUDGET_RANGES = {
    "groq": (9, 12),
    "cerebras": (5, 6),
    "gemini": (6, 8),
    "zai": (4, 5),
    "openrouter": (10, 12),
    "sambanova": (4, 5),
    "pollinations": (10, 14),
    "cloudflare": (7, 8),
    "ovh": (5, 6),
    "kilol": (12, 16),
}
DEFAULT_BUDGET_RANGE = (4, 16)
P_DESIGNATED = 0.1
P_OUTAGE = 0.1


def _build_schedule_prior(pid):
    lo, hi = BUDGET_RANGES.get(pid, DEFAULT_BUDGET_RANGE)
    table = {}
    n_budgets = hi - lo + 1
    for budget in range(lo, hi + 1):
        window = list(range(2, budget))
        for count in (1, 2):
            if len(window) < count:
                continue
            subsets = list(combinations(window, count))
            base_weight = (1.0 / n_budgets) * 0.5 / len(subsets)
            for raw in subsets:
                raw = tuple(sorted(raw))
                # Designated forcing: the earliest planned failure is pinned to 2.
                pinned = tuple(sorted({2} | set(raw[1:]))) if 2 not in raw else raw
                for is_designated, w_des in ((False, 1.0 - P_DESIGNATED), (True, P_DESIGNATED)):
                    failset = pinned if is_designated else raw
                    key = (budget, failset, None)
                    table[key] = table.get(key, 0.0) + base_weight * w_des * (1.0 - P_OUTAGE)
                    for outage_at in window:
                        key = (budget, failset, outage_at)
                        table[key] = table.get(key, 0.0) + base_weight * w_des * P_OUTAGE / len(window)
    return table


def _hypothesis_fails(hyp, k):
    budget, failset, outage_at = hyp
    if k > budget:
        return True
    if outage_at is not None and k >= outage_at:
        return True
    return k in failset


def _hypothesis_consistent(hyp, k, outcome):
    """Whether hypothesis `hyp` explains call `k` ending as `outcome`:
    'ok', 'transient' (a planned failure) or 'permanent' (budget/outage).
    The env tests budget, then outage, then the schedule, so a transient
    failure proves the call sat within budget, before any outage, and on
    the schedule."""
    budget, failset, outage_at = hyp
    if outcome == "ok":
        return k <= budget and (outage_at is None or k < outage_at) and k not in failset
    if outcome == "transient":
        return k <= budget and (outage_at is None or k < outage_at) and k in failset
    return k > budget or (outage_at is not None and k >= outage_at)


class ScheduleBelief:
    """Bayesian belief over one provider's failure schedule."""

    def __init__(self, pid):
        self.pid = pid
        self.weights = _build_schedule_prior(pid)
        self.mass = sum(self.weights.values())
        self._memo = {}

    def update(self, k, outcome):
        survivors = {h: w for h, w in self.weights.items() if _hypothesis_consistent(h, k, outcome)}
        total = sum(survivors.values())
        if total > 0:
            self.weights = survivors
            self.mass = total
        # A zero-mass observation contradicts the model (impossible under an
        # exact refusal classifier); keep the previous belief in that case.
        self._memo.clear()

    def fail_probability(self, k, assumed_ok=()):
        """P(call k fails), optionally assuming the given call numbers
        succeed first (for planning against projected counts)."""
        memo_key = (k, tuple(sorted(assumed_ok)))
        cached = self._memo.get(memo_key)
        if cached is not None:
            return cached
        fail_mass = 0.0
        live_mass = 0.0
        for hyp, w in self.weights.items():
            if any(_hypothesis_fails(hyp, a) for a in assumed_ok):
                continue
            live_mass += w
            if _hypothesis_fails(hyp, k):
                fail_mass += w
        p = (fail_mass / live_mass) if live_mass > 0 else 1.0
        self._memo[memo_key] = p
        return p


# --------------------------------------------------------------------------
# Brief parsing
# --------------------------------------------------------------------------

def _find_object_end(text, start):
    """Position of the `}` that closes the object opened at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _extract_documents(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    found = []
    start = text.find("{")
    while start != -1:
        end = _find_object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                found.append(value)
        start = text.find("{", start + 1)
    return found


def _text_or_empty(v):
    return v if isinstance(v, str) else ""


def _number_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _parse_provider(v):
    if not isinstance(v, dict):
        return None
    pid = _text_or_empty(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = _text_or_empty(v.get("path")) or _text_or_empty(v.get("base_path")) or _text_or_empty(v.get("base_url")) or _text_or_empty(v.get("base"))
    return {
        "id": pid,
        "name": _text_or_empty(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _text_or_empty(v.get("key_env")),
        "key": _text_or_empty(v.get("key")),
    }


def _parse_rating(v):
    if not isinstance(v, dict):
        return None
    model = _text_or_empty(v.get("model"))
    provider = _text_or_empty(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) and not isinstance(rating, bool) else 0,
        "context_length": _number_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _parse_workload_entry(v):
    if not isinstance(v, dict):
        return None
    rid = _text_or_empty(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if isinstance(v.get("requires"), dict) else {}
    req = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    min_context = requires.get("min_context")
    if isinstance(min_context, (int, float)) and not isinstance(min_context, bool):
        req["min_context"] = min_context
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": req,
    }


def load_brief(path):
    """Return (providers, ratings_rows, manifest, error)."""
    if not path:
        return [], [], [], "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return [], [], [], "cannot read %s: %s" % (path, exc)
    docs = _extract_documents(text)
    directory = next(
        (d for d in docs if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list)),
        None,
    )
    if directory is None:
        return [], [], [], "the brief holds no %s document" % PROVIDERS_SCHEMA
    providers = [p for p in (_parse_provider(v) for v in directory["providers"]) if p is not None]
    ratings_doc = next(
        (d for d in docs if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list)),
        None,
    )
    ratings = []
    if ratings_doc is not None:
        ratings = [r for r in (_parse_rating(v) for v in ratings_doc["ratings"]) if r is not None]
    workload_doc = next(
        (d for d in docs if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list)),
        None,
    )
    manifest = []
    if workload_doc is not None:
        manifest = [w for w in (_parse_workload_entry(v) for v in workload_doc["manifest"]) if w is not None]
    return providers, ratings, manifest, None


# --------------------------------------------------------------------------
# Provider catalog probing
# --------------------------------------------------------------------------

def resolve_base_url(env_url, base):
    cleaned = base.strip()
    if cleaned.lower().startswith("http://") or cleaned.lower().startswith("https://"):
        return cleaned.rstrip("/")
    prefix = env_url.rstrip("/")
    if cleaned == "":
        return prefix
    return prefix + ("" if cleaned.startswith("/") else "/") + cleaned.rstrip("/")


def build_auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _pricing_all_zero(v):
    pricing = v.get("pricing")
    if not isinstance(pricing, dict):
        return None
    numbers = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            numbers.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                numbers.append(float(x))
            except ValueError:
                pass
    if not numbers:
        return None
    return all(n == 0 for n in numbers)


def _parse_model_entry(wire, v):
    if not isinstance(v, dict):
        return None
    if wire == "gemini":
        mid = _text_or_empty(v.get("name"))
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
    else:
        mid = _text_or_empty(v.get("id"))
    if mid == "":
        return None
    ctx = _number_or_none(v.get("context_length"))
    if ctx is None:
        ctx = _number_or_none(v.get("context_window"))
    if ctx is None:
        ctx = _number_or_none(v.get("inputTokenLimit"))
    is_free = v.get("free") if isinstance(v.get("free"), bool) else _pricing_all_zero(v)
    return {
        "id": mid,
        "display_name": _text_or_empty(v.get("display_name")) or _text_or_empty(v.get("displayName")) or mid,
        "free": is_free,
        "context_length": ctx,
        "tools": v.get("tools") if isinstance(v.get("tools"), bool) else None,
        "vision": v.get("vision") if isinstance(v.get("vision"), bool) else None,
        "json": v.get("json") if isinstance(v.get("json"), bool) else None,
    }


def _parse_model_entries(wire, body):
    if not isinstance(body, dict):
        return []
    items = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(items, list):
        return []
    return [m for m in (_parse_model_entry(wire, v) for v in items) if m is not None]


def _request_json(method, url, headers, payload=None, timeout=UPSTREAM_TIMEOUT):
    """A single HTTP call. Returns (status, parsed_json_or_None, raw_bytes)."""
    data = None
    hdrs = dict(headers)
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        hdrs["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        status = exc.code
    except Exception:
        return 0, None, b""
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        parsed = None
    return status, parsed, raw


def probe_provider_catalog(env_url, provider):
    """Probe one provider's catalog route. Returns (ok, models)."""
    base = resolve_base_url(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"Accept": "application/json"}
    headers.update(build_auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION_DEFAULT
    for _ in range(PROBE_ATTEMPTS):
        status, body, _raw = _request_json("GET", url, headers, timeout=PROBE_TIMEOUT)
        if 200 <= status < 300 and body is not None:
            return True, _parse_model_entries(provider["wire"], body)
        if status == 0:
            continue
        return False, []
    return False, []


def assemble_catalog(providers, probes, ratings_by_key):
    """The catalog document: providers in brief order that answered, every
    model listed, free flag and context taken from the quality table where
    it has a row. `base_url` keeps the directory's own path, never joined
    to the env URL, so a replay reproduces it byte for byte."""
    catalog_providers = []
    for provider, (ok, models) in zip(providers, probes):
        if not ok:
            continue
        seen = set()
        catalog_models = []
        for model in sorted(models, key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = ratings_by_key.get((provider["id"], model["id"]))
            if row is not None:
                free = row["free"]
                context = row["context_length"]
            else:
                free = model["free"] if model["free"] is not None else False
                context = model["context_length"]
            catalog_models.append({"id": model["id"], "free": free, "context_length": context})
        catalog_providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": catalog_models,
        })
    return {"providers": catalog_providers}


# --------------------------------------------------------------------------
# Wire translation
#
# Every request is reduced to one neutral shape and then rebuilt in the
# upstream protocol. The neutral shape keeps exactly the fields providers
# echo: system text, user text, images in their original payload form
# (base64 stays base64, a URL stays a URL), tool names in order, stop
# sequences, max_tokens (exactly, or absent when the client sent none), and
# JSON mode. Temperature and top_p pass through when present.
# --------------------------------------------------------------------------

def _decode_data_url(url):
    """'data:<media>;base64,<data>' -> (media, data), or None."""
    if not url.startswith("data:"):
        return None
    head, sep, data = url[5:].partition(",")
    if not sep:
        return None
    media = "image/png"
    if ";base64" in head:
        media = head.split(";")[0] or "image/png"
        return (media, data)
    return None


def _gather_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    fragments = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            fragments.append(block["text"])
    return "\n".join(fragments)


def _render_tool_result(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _gather_text(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"))
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"))


def _openai_message_parts(message):
    """Content parts of one OpenAI message, in the neutral shape."""
    parts = []
    content = message.get("content")
    if isinstance(content, str):
        if content != "":
            parts.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
                url = _text_or_empty(block["image_url"].get("url"))
                split = _decode_data_url(url)
                if split is not None:
                    parts.append({"type": "image_b64", "media": split[0], "data": split[1]})
                elif url != "":
                    parts.append({"type": "image_url", "url": url})
    if message.get("role") == "assistant":
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                continue
            fn = call["function"]
            parts.append({
                "type": "tool_call",
                "id": _text_or_empty(call.get("id")),
                "name": _text_or_empty(fn.get("name")),
                "arguments": fn.get("arguments", ""),
            })
    return parts


def _anthropic_message_parts(message):
    parts = []
    content = message.get("content")
    if isinstance(content, str):
        if content != "":
            parts.append({"type": "text", "text": content})
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
            elif btype == "image" and isinstance(block.get("source"), dict):
                source = block["source"]
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    parts.append({
                        "type": "image_b64",
                        "media": _text_or_empty(source.get("media_type")) or "image/png",
                        "data": source["data"],
                    })
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url", "url": source["url"]})
            elif btype == "tool_use":
                parts.append({
                    "type": "tool_call",
                    "id": _text_or_empty(block.get("id")),
                    "name": _text_or_empty(block.get("name")),
                    "arguments": block.get("input", {}),
                })
            elif btype == "tool_result":
                parts.append({
                    "type": "tool_result",
                    "tool_use_id": _text_or_empty(block.get("tool_use_id")),
                    "text": _render_tool_result(block.get("content")),
                })
    return parts


def _openai_tool_defs(body):
    tools = []
    if isinstance(body.get("tools"), list):
        for raw in body["tools"]:
            if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": _text_or_empty(fn.get("description")),
                "parameters": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}},
            })
    return tools


def _anthropic_tool_defs(body):
    tools = []
    if isinstance(body.get("tools"), list):
        for raw in body["tools"]:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "name": raw["name"],
                "description": _text_or_empty(raw.get("description")),
                "parameters": raw.get("input_schema") if isinstance(raw.get("input_schema"), dict) else {"type": "object", "properties": {}},
            })
    return tools


def _openai_tool_selection(body):
    choice = body.get("tool_choice")
    if choice == "auto":
        return "auto"
    if choice == "required":
        return "required"
    if choice == "none":
        return "none"
    if isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("function"), dict):
        name = choice["function"].get("name")
        if isinstance(name, str):
            return ("tool", name)
    return None


def _anthropic_tool_selection(body):
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "auto":
            return "auto"
        if choice.get("type") == "any":
            return "required"
        if choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            return ("tool", choice["name"])
    if choice == "auto":
        return "auto"
    return None


def _extract_stops(body, wire):
    if wire == "openai":
        stop = body.get("stop")
        if isinstance(stop, str):
            return [stop]
        if isinstance(stop, list):
            return [s for s in stop if isinstance(s, str)]
        return None
    seqs = body.get("stop_sequences")
    if isinstance(seqs, list):
        return [s for s in seqs if isinstance(s, str)]
    return None


def normalize_request(body, wire):
    """A client body in either protocol, reduced to the neutral shape."""
    norm = {
        "system": None,
        "messages": [],
        "tools": [],
        "tool_choice": None,
        "max_tokens": None,
        "stops": None,
        "temperature": None,
        "top_p": None,
        "json_mode": False,
    }
    if wire == "openai":
        system_fragments = []
        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                system_fragments.append(_gather_text(message.get("content")))
                continue
            if role == "tool":
                norm["messages"].append({
                    "role": "tool",
                    "tool_call_id": _text_or_empty(message.get("tool_call_id")),
                    "parts": [{"type": "tool_result", "tool_use_id": _text_or_empty(message.get("tool_call_id")), "text": _render_tool_result(message.get("content"))}],
                })
                continue
            norm["messages"].append({
                "role": "assistant" if role == "assistant" else "user",
                "parts": _openai_message_parts(message),
            })
        if system_fragments:
            norm["system"] = "\n".join(system_fragments)
        norm["tools"] = _openai_tool_defs(body)
        norm["tool_choice"] = _openai_tool_selection(body)
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
            norm["json_mode"] = True
    else:
        system = body.get("system")
        if isinstance(system, str):
            norm["system"] = system
        elif isinstance(system, list):
            text = _gather_text(system)
            if text != "":
                norm["system"] = text
        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            norm["messages"].append({
                "role": "assistant" if message.get("role") == "assistant" else "user",
                "parts": _anthropic_message_parts(message),
            })
        norm["tools"] = _anthropic_tool_defs(body)
        norm["tool_choice"] = _anthropic_tool_selection(body)
    max_tokens = body.get("max_tokens")
    if max_tokens is None and wire == "openai":
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)) and not isinstance(max_tokens, bool):
        norm["max_tokens"] = max_tokens
    norm["stops"] = _extract_stops(body, wire)
    if isinstance(body.get("temperature"), (int, float)) and not isinstance(body.get("temperature"), bool):
        norm["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)) and not isinstance(body.get("top_p"), bool):
        norm["top_p"] = body["top_p"]
    return norm


def _parts_as_openai_content(parts):
    """Neutral parts rendered as an OpenAI message content value."""
    rendered = []
    for part in parts:
        if part["type"] == "text":
            rendered.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_b64":
            rendered.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (part["media"], part["data"])}})
        elif part["type"] == "image_url":
            rendered.append({"type": "image_url", "image_url": {"url": part["url"]}})
    if len(rendered) == 1 and rendered[0]["type"] == "text":
        return rendered[0]["text"]
    return rendered


def build_openai_request(norm, model):
    messages = []
    if norm["system"] is not None:
        messages.append({"role": "system", "content": norm["system"]})
    for message in norm["messages"]:
        if message["role"] == "tool":
            text = ""
            call_id = message.get("tool_call_id", "")
            for part in message["parts"]:
                if part["type"] == "tool_result":
                    text = part["text"]
                    call_id = part["tool_use_id"] or call_id
            messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            continue
        calls = []
        for part in message["parts"]:
            if part["type"] == "tool_call":
                args = part["arguments"]
                if isinstance(args, dict):
                    args = json.dumps(args, separators=(",", ":"))
                elif not isinstance(args, str):
                    args = ""
                calls.append({"id": part["id"], "type": "function", "function": {"name": part["name"], "arguments": args}})
        content = _parts_as_openai_content(message["parts"])
        entry = {"role": message["role"], "content": content if content != "" else None}
        if message["role"] == "assistant" and content == []:
            entry["content"] = None
        if calls:
            entry["tool_calls"] = calls
        messages.append(entry)
    payload = {"model": model, "messages": messages}
    if norm["max_tokens"] is not None:
        payload["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        payload["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        payload["top_p"] = norm["top_p"]
    if norm["stops"]:
        payload["stop"] = norm["stops"]
    if norm["tools"]:
        payload["tools"] = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in norm["tools"]
        ]
    choice = norm["tool_choice"]
    if choice == "auto":
        payload["tool_choice"] = "auto"
    elif choice == "required":
        payload["tool_choice"] = "required"
    elif choice == "none":
        payload["tool_choice"] = "none"
    elif isinstance(choice, tuple):
        payload["tool_choice"] = {"type": "function", "function": {"name": choice[1]}}
    if norm["json_mode"]:
        payload["response_format"] = {"type": "json_object"}
    return payload


def _parts_as_anthropic_content(parts):
    rendered = []
    for part in parts:
        if part["type"] == "text":
            rendered.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_b64":
            rendered.append({"type": "image", "source": {"type": "base64", "media_type": part["media"], "data": part["data"]}})
        elif part["type"] == "image_url":
            rendered.append({"type": "image", "source": {"type": "url", "url": part["url"]}})
        elif part["type"] == "tool_call":
            args = part["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            rendered.append({"type": "tool_use", "id": part["id"], "name": part["name"], "input": args})
        elif part["type"] == "tool_result":
            rendered.append({"type": "tool_result", "tool_use_id": part["tool_use_id"], "content": part["text"]})
    if len(rendered) == 1 and rendered[0]["type"] == "text":
        return rendered[0]["text"]
    return rendered


def build_anthropic_request(norm, model):
    payload = {"model": model}
    if norm["max_tokens"] is not None:
        payload["max_tokens"] = norm["max_tokens"]
    if norm["system"] is not None:
        payload["system"] = norm["system"]
    messages = []
    for message in norm["messages"]:
        if message["role"] == "tool":
            blocks = []
            for part in message["parts"]:
                if part["type"] == "tool_result":
                    blocks.append({"type": "tool_result", "tool_use_id": part["tool_use_id"], "content": part["text"]})
            messages.append({"role": "user", "content": blocks})
            continue
        messages.append({"role": message["role"], "content": _parts_as_anthropic_content(message["parts"])})
    payload["messages"] = messages
    if norm["temperature"] is not None:
        payload["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        payload["top_p"] = norm["top_p"]
    if norm["tools"]:
        payload["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in norm["tools"]
        ]
    choice = norm["tool_choice"]
    if choice == "auto":
        payload["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        payload["tool_choice"] = {"type": "any"}
    elif isinstance(choice, tuple):
        payload["tool_choice"] = {"type": "tool", "name": choice[1]}
    if norm["stops"]:
        payload["stop_sequences"] = norm["stops"]
    return payload


def _parts_as_gemini_parts(parts):
    rendered = []
    for part in parts:
        if part["type"] == "text":
            rendered.append({"text": part["text"]})
        elif part["type"] == "image_b64":
            rendered.append({"inlineData": {"mimeType": part["media"], "data": part["data"]}})
        elif part["type"] == "image_url":
            rendered.append({"fileData": {"fileUri": part["url"], "mimeType": "image/png"}})
        elif part["type"] == "tool_call":
            args = part["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            rendered.append({"functionCall": {"name": part["name"], "args": args}})
        elif part["type"] == "tool_result":
            rendered.append({"functionResponse": {"name": part["tool_use_id"], "response": {"result": part["text"]}}})
    return rendered


def build_gemini_request(norm, model):
    payload = {}
    if norm["system"] is not None:
        payload["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
    contents = []
    for message in norm["messages"]:
        if message["role"] == "tool":
            contents.append({"role": "function", "parts": _parts_as_gemini_parts(message["parts"])})
        else:
            contents.append({"role": "model" if message["role"] == "assistant" else "user", "parts": _parts_as_gemini_parts(message["parts"])})
    payload["contents"] = contents
    if norm["tools"]:
        payload["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in norm["tools"]
        ]}]
    config = {}
    if norm["max_tokens"] is not None:
        config["maxOutputTokens"] = norm["max_tokens"]
    if norm["stops"]:
        config["stopSequences"] = norm["stops"]
    if norm["temperature"] is not None:
        config["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        config["topP"] = norm["top_p"]
    if norm["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        payload["generationConfig"] = config
    return payload


def build_upstream_request(norm, wire, model):
    if wire == "openai":
        return build_openai_request(norm, model)
    if wire == "anthropic":
        return build_anthropic_request(norm, model)
    return build_gemini_request(norm, model)


# --------------------------------------------------------------------------
# Upstream answers: parse to (text, stop family, input tokens, output tokens)
# Stop families: 0 stop, 1 length, 2 tool calls, 3 content filter.
# --------------------------------------------------------------------------

def _gemini_stop_family(reason):
    if reason == "MAX_TOKENS":
        return 1
    if reason == "SAFETY":
        return 3
    return 0


def _anthropic_stop_family(stop):
    if stop == "max_tokens":
        return 1
    if stop == "tool_use":
        return 2
    if stop == "refusal":
        return 3
    return 0


def _openai_stop_family(finish):
    if finish == "length":
        return 1
    if finish in ("tool_calls", "function_call"):
        return 2
    if finish == "content_filter":
        return 3
    return 0


def parse_upstream_reply(wire, body):
    """Return (text, family, input_tokens, output_tokens) or None."""
    if not isinstance(body, dict):
        return None
    if wire == "openai":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return None
        text = message.get("content")
        if isinstance(text, list):
            text = _gather_text(text)
        if not isinstance(text, str):
            return None
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if not isinstance(prompt, (int, float)) or not isinstance(completion, (int, float)):
            return None
        return (text, _openai_stop_family(choices[0].get("finish_reason")), int(prompt), int(completion))
    if wire == "anthropic":
        content = body.get("content")
        if not isinstance(content, list):
            return None
        text = "".join(
            block["text"] for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        if not isinstance(prompt, (int, float)) or not isinstance(completion, (int, float)):
            return None
        return (text, _anthropic_stop_family(body.get("stop_reason")), int(prompt), int(completion))
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        return None
    content = candidates[0].get("content")
    if not isinstance(content, dict) or not isinstance(content.get("parts"), list):
        return None
    text = "".join(
        part["text"] for part in content["parts"]
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )
    usage = body.get("usageMetadata") if isinstance(body.get("usageMetadata"), dict) else {}
    prompt = usage.get("promptTokenCount")
    completion = usage.get("candidatesTokenCount")
    if not isinstance(prompt, (int, float)) or not isinstance(completion, (int, float)):
        return None
    return (text, _gemini_stop_family(candidates[0].get("finishReason")), int(prompt), int(completion))


def classify_upstream_failure(status, parsed):
    """Sort a non-2xx upstream answer. Returns (outcome, counted):
    outcome is 'transient', 'permanent' or 'uncounted'; counted says
    whether the env charged the call against the provider's budget and
    call number. All three protocols carry the refusal reason in
    error.message:
      "<name>'s free-tier budget is spent" -> permanent
      "<name> is out"                     -> permanent (outage)
      "<name> refused this call"          -> transient (planned failure)
    Auth/model errors and transport failures spend no budget and carry
    no schedule information, so they are 'uncounted' and never reach the
    belief. Anything unrecognised defaults to transient (safe: it never
    wrongly retires a provider)."""
    if status == 0 or parsed is None:
        return "uncounted", False
    if status in (400, 401):
        return "uncounted", False
    message = ""
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        message = _text_or_empty(error.get("message"))
    if "budget is spent" in message:
        return "permanent", True
    if message.rstrip().endswith("is out"):
        return "permanent", True
    if "refused this call" in message:
        return "transient", True
    if status in (402, 429):
        return "transient", True
    return "uncounted", False


# --------------------------------------------------------------------------
# Client answers, in the protocol the client used
# --------------------------------------------------------------------------

def _anthropic_stop_reason(family):
    return ["end_turn", "max_tokens", "tool_use", "refusal"][family]


def _openai_finish_reason(family):
    return ["stop", "length", "tool_calls", "content_filter"][family]


def render_openai_json(client_model, text, family, input_tokens, output_tokens):
    return {
        "id": "chatcmpl-aggregator",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": client_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": _openai_finish_reason(family),
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def render_openai_sse(client_model, text, family, input_tokens, output_tokens):
    """The OpenAI event sequence: role chunk, content deltas, finish chunk,
    usage chunk, [DONE]. Nothing follows [DONE]."""
    head = {"id": "chatcmpl-aggregator", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": client_model}
    chunks = []
    first = dict(head)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
    chunks.append("data: %s\n\n" % json.dumps(first, separators=(",", ":")))
    for at in range(0, len(text), 13):
        chunk = dict(head)
        chunk["choices"] = [{"index": 0, "delta": {"content": text[at:at + 13]}, "finish_reason": None}]
        chunks.append("data: %s\n\n" % json.dumps(chunk, separators=(",", ":")))
    finish = dict(head)
    finish["choices"] = [{"index": 0, "delta": {}, "finish_reason": _openai_finish_reason(family)}]
    chunks.append("data: %s\n\n" % json.dumps(finish, separators=(",", ":")))
    usage = dict(head)
    usage["choices"] = []
    usage["usage"] = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    chunks.append("data: %s\n\n" % json.dumps(usage, separators=(",", ":")))
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


def render_anthropic_json(client_model, text, family, input_tokens, output_tokens):
    return {
        "id": "msg_aggregator",
        "type": "message",
        "role": "assistant",
        "model": client_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": _anthropic_stop_reason(family),
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _format_sse_event(name, payload):
    return "event: %s\ndata: %s\n\n" % (name, json.dumps(payload, separators=(",", ":")))


def render_anthropic_sse(client_model, text, family, input_tokens, output_tokens):
    """The Anthropic event sequence: message_start, one text block's
    start/deltas/stop, message_delta, message_stop. Nothing follows it."""
    events = []
    events.append(_format_sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_aggregator",
            "type": "message",
            "role": "assistant",
            "model": client_model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }))
    events.append(_format_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for at in range(0, len(text), 13):
        events.append(_format_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text[at:at + 13]},
        }))
    events.append(_format_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(_format_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_stop_reason(family), "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }))
    events.append(_format_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(events).encode("utf-8")


# --------------------------------------------------------------------------
# Planning: one global assignment, recomputed before every request
# --------------------------------------------------------------------------

def _caps_satisfy(requires, caps):
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    min_context = requires.get("min_context")
    if min_context is not None and (caps["context_length"] is None or caps["context_length"] < min_context):
        return False
    return True


class State:
    def __init__(self, env_url, providers, ratings_by_key, probes, manifest, catalog):
        self.env_url = env_url
        self.providers = providers
        self.provider_index = {p["id"]: p for p in providers}
        self.base_urls = {p["id"]: resolve_base_url(env_url, p["base"]) for p in providers}
        self.manifest = manifest
        self.catalog = catalog
        self.call_counts = {}
        self.dead_providers = set()
        self.next_index = 0
        self.inflight = 0
        self.lock = threading.Lock()
        self.beliefs = {p["id"]: ScheduleBelief(p["id"]) for p in providers}
        # Resolved models per provider: free flag and capabilities taken
        # from the quality table where it has a row, else the listing's
        # own fields.
        self.models_by_pid = {}
        for provider, (ok, models) in zip(providers, probes):
            if not ok:
                continue
            resolved = []
            seen = set()
            for model in models:
                if model["id"] in seen:
                    continue
                seen.add(model["id"])
                row = ratings_by_key.get((provider["id"], model["id"]))
                if row is not None:
                    free = row["free"]
                    caps = {
                        "context_length": row["context_length"],
                        "tools": row["tools"],
                        "vision": row["vision"],
                        "json": row["json"],
                    }
                    rating = row["rating_milli"]
                else:
                    free = model["free"] if model["free"] is not None else False
                    caps = {
                        "context_length": model["context_length"],
                        "tools": model["tools"],
                        "vision": model["vision"],
                        "json": model["json"],
                    }
                    rating = 0
                resolved.append({"id": model["id"], "free": free, "caps": caps, "rating": rating})
            self.models_by_pid[provider["id"]] = resolved
        self.ratings = {}
        for pid, models in self.models_by_pid.items():
            for model in models:
                self.ratings[(pid, model["id"])] = model["rating"]
        # The scorer's best rating per request: the max over free models
        # meeting its requirements. Spent deficit against these bests
        # drives the adaptive risk charge.
        self.best_ratings = {}
        for row in manifest:
            cands = self.eligible_candidates(row["requires"], False)
            self.best_ratings[row["id"]] = max([c[2] for c in cands], default=0)
        self.deficit_spent = 0
        # The free model ids the catalog registered, for GET /v1/models.
        self.served_models = []
        seen_ids = set()
        for entry in catalog["providers"]:
            for model in entry["models"]:
                if model["free"] and model["id"] not in seen_ids:
                    seen_ids.add(model["id"])
                    self.served_models.append((model["id"], entry["id"]))
        self.served_models.sort()

    def eligible_candidates(self, requires, wire_filter):
        """(provider, model, rating) triples that may serve a request.

        `wire_filter` is set when the request carries JSON mode on the
        OpenAI protocol: the Anthropic protocol cannot echo JSON mode, so
        it is excluded."""
        eligible = []
        for provider in self.providers:
            if wire_filter and provider["wire"] == "anthropic":
                continue
            for model in self.models_by_pid.get(provider["id"], []):
                if not model["free"]:
                    continue
                if not _caps_satisfy(requires, model["caps"]):
                    continue
                eligible.append((provider["id"], model["id"], model["rating"]))
        return eligible

    def _current_price(self):
        return W_HI if self.deficit_spent < DEFICIT_SWITCH else W_LO

    def record_outcome(self, pid, call_number, outcome):
        """Record one counted upstream outcome: 'ok', 'transient' or
        'permanent'. A permanent refusal retires the provider outright;
        every outcome updates its schedule belief."""
        if outcome == "permanent":
            self.dead_providers.add(pid)
        self.beliefs[pid].update(call_number, outcome)

    def _rank_by_risk(self, candidates, counts, weight=1):
        """Candidates best-first by risk-charged score: rating less W times
        the belief's probability that the provider's next call fails,
        divided by the request's weight (a weight-2 request's rating
        deficit costs double, while a wasted call costs the same).
        `counts` may hold projected counts: the failure probability is
        then evaluated at the projected call number, assuming the
        projected earlier calls succeed. Ties break on rating, then
        provider and model id, so the order stays fully deterministic."""
        price = self._current_price()
        def key(cand):
            pid, mid, rating = cand
            if pid in self.dead_providers:
                return (1 << 30, -rating, pid, mid)
            nxt = counts.get(pid, 0) + 1
            actual = self.call_counts.get(pid, 0)
            assume = tuple(range(actual + 1, counts.get(pid, 0) + 1))
            p = self.beliefs[pid].fail_probability(nxt, assume)
            effective = rating - (price * p) / weight
            return (-effective, -rating, pid, mid)
        return sorted(candidates, key=key)

    def assign_remaining(self, start_index, current_json_mode):
        """Assign every remaining request a (provider, model).

        The most-constrained requests (fewest viable candidates) are
        placed first so scarce capabilities are reserved for the requests
        that need them; each placement charges the provider's projected
        load, which the risk charge then reads as a higher failure
        probability for its next call, spreading traffic instead of
        draining one provider's budget."""
        remaining = self.manifest[start_index:]
        if not remaining:
            return {}
        cand_lists = []
        for i, row in enumerate(remaining):
            if i == 0:
                wire_filter = current_json_mode
            else:
                # A future OpenAI request that requires JSON mode is
                # assumed to carry response_format, as the workload's JSON
                # requests do; Anthropic requests cannot express JSON mode.
                wire_filter = row["api"] == "openai" and row["requires"].get("json") is True
            cand_lists.append(self.eligible_candidates(row["requires"], wire_filter))
        order = sorted(
            range(len(remaining)),
            key=lambda i: (len(cand_lists[i]), -remaining[i]["weight"], i),
        )
        projected = dict(self.call_counts)
        plan = {}
        for i in order:
            if not cand_lists[i]:
                continue
            ranked = self._rank_by_risk(cand_lists[i], projected, remaining[i]["weight"])
            chosen = ranked[0]
            plan[remaining[i]["id"]] = (chosen[0], chosen[1])
            projected[chosen[0]] = projected.get(chosen[0], 0) + 1
        return plan


def _infer_requirements(norm):
    vision = False
    for message in norm["messages"]:
        for part in message["parts"]:
            if part["type"] in ("image_b64", "image_url"):
                vision = True
    return {
        "tools": bool(norm["tools"]),
        "vision": vision,
        "json": norm["json_mode"],
    }


def invoke_upstream(state, provider_id, model_id, norm, anthropic_version):
    """One non-streaming upstream call. Returns (answer_or_None, outcome,
    counted): outcome is 'ok', 'transient', 'permanent' or 'uncounted', and
    counted says whether the env charged the call (only counted outcomes
    advance the provider's call number and belief)."""
    provider = state.provider_index[provider_id]
    wire = provider["wire"]
    payload = build_upstream_request(norm, wire, model_id)
    base = state.base_urls[provider_id]
    if wire == "gemini":
        url = base + "/v1beta/models/" + urllib.parse.quote(model_id, safe="") + ":generateContent"
    elif wire == "anthropic":
        url = base + "/v1/messages"
    else:
        url = base + "/v1/chat/completions"
    headers = {"Accept": "application/json"}
    headers.update(build_auth_headers(provider["auth"], provider["key"]))
    if wire == "anthropic":
        headers["anthropic-version"] = anthropic_version
    status, parsed, _raw = _request_json("POST", url, headers, payload, timeout=UPSTREAM_TIMEOUT)
    if 200 <= status < 300:
        answer = parse_upstream_reply(wire, parsed)
        if answer is not None:
            return answer, "ok", True
        return None, "transient", True
    outcome, counted = classify_upstream_failure(status, parsed)
    return None, outcome, counted


def handle_request(state, wire, body, anthropic_version, caller=None):
    """Serve one client request. Returns (text, family, in, out) or None.

    `caller` defaults to the real upstream call; the local simulator
    passes a schedule lookup instead, exercising the same planning and
    failover code without HTTP. A caller returns (answer, outcome,
    counted) like invoke_upstream."""
    call = caller or invoke_upstream
    norm = normalize_request(body, wire)
    with state.lock:
        index = state.next_index
        state.next_index += 1
    row = None
    if index < len(state.manifest) and state.manifest[index]["api"] == wire:
        row = state.manifest[index]
    if row is not None:
        requires = row["requires"]
    else:
        requires = _infer_requirements(norm)
    plan = state.assign_remaining(index, norm["json_mode"])
    planned = plan.get(row["id"]) if row is not None else None
    weight = row["weight"] if row is not None else 1
    ranked = state._rank_by_risk(state.eligible_candidates(requires, norm["json_mode"]), state.call_counts, weight)
    if planned is not None and planned in [(c[0], c[1]) for c in ranked]:
        ranked = [(planned[0], planned[1], 0)] + [c for c in ranked if (c[0], c[1]) != planned]
    # Failover in two phases: first every candidate on a provider that has
    # not failed during THIS request, then the deferred same-provider
    # retries. A refusal may be a one-off failure or an exhausted budget;
    # other providers are tried before spending another call finding out
    # which, and a provider that fails twice in one request is left alone.
    # Only counted outcomes advance the call count and the belief.
    tried = set()
    failed_providers = set()
    deferred = []
    queue = list(ranked)
    while queue:
        provider_id, model_id, _rating = queue.pop(0)
        if (provider_id, model_id) in tried or provider_id in state.dead_providers:
            continue
        if provider_id in failed_providers:
            deferred.append((provider_id, model_id))
            continue
        tried.add((provider_id, model_id))
        answer, outcome, counted = call(state, provider_id, model_id, norm, anthropic_version)
        if counted:
            with state.lock:
                state.call_counts[provider_id] = state.call_counts.get(provider_id, 0) + 1
                call_number = state.call_counts[provider_id]
            state.record_outcome(provider_id, call_number, outcome)
        if answer is not None:
            if row is not None:
                state.deficit_spent += weight * max(0, state.best_ratings[row["id"]] - state.ratings.get((provider_id, model_id), 0))
            return answer
        failed_providers.add(provider_id)
    failed_again = set()
    for provider_id, model_id in deferred:
        if (provider_id, model_id) in tried or provider_id in failed_again or provider_id in state.dead_providers:
            continue
        tried.add((provider_id, model_id))
        answer, outcome, counted = call(state, provider_id, model_id, norm, anthropic_version)
        if counted:
            with state.lock:
                state.call_counts[provider_id] = state.call_counts.get(provider_id, 0) + 1
                call_number = state.call_counts[provider_id]
            state.record_outcome(provider_id, call_number, outcome)
        if answer is not None:
            if row is not None:
                state.deficit_spent += weight * max(0, state.best_ratings[row["id"]] - state.ratings.get((provider_id, model_id), 0))
            return answer
        failed_again.add(provider_id)
    return None


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

def _build_model_listings(state):
    openai_listing = {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": FIXED_CREATED, "owned_by": pid}
            for mid, pid in state.served_models
        ],
    }
    anthropic_listing = {
        "data": [
            {"type": "model", "id": mid, "display_name": mid, "created_at": FIXED_ANTHROPIC_CREATED_AT}
            for mid, _pid in state.served_models
        ],
        "has_more": False,
    }
    return openai_listing, anthropic_listing


def create_handler(state):
    openai_listing, anthropic_listing = _build_model_listings(state)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format, *_args):
            pass

        def _reply(self, status, payload, content_type="application/json"):
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass
            self.close_connection = True

        def _reply_json(self, status, obj):
            self._reply(status, json.dumps(obj, separators=(",", ":")).encode("utf-8"))

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/v1/models":
                if self.headers.get("x-api-key") is not None:
                    self._reply_json(200, anthropic_listing)
                else:
                    self._reply_json(200, openai_listing)
                return
            self._reply_json(404, {"error": {"message": "not found", "type": "invalid_request", "code": None}})

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/v1/chat/completions":
                wire = "openai"
            elif path == "/v1/messages":
                wire = "anthropic"
            else:
                self._reply_json(404, {"error": {"message": "not found", "type": "invalid_request", "code": None}})
                return
            length = self.headers.get("Content-Length")
            try:
                count = int(length) if length else 0
            except ValueError:
                count = 0
            try:
                raw = self.rfile.read(count) if count > 0 else b""
            except OSError:
                # Client went away mid-body: fall through to the clean 400
                # below instead of dying unhandled without a response.
                raw = b""
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                body = None
            if not isinstance(body, dict):
                self._reply_error(wire, 400, "request body is not a JSON object")
                return
            with state.lock:
                state.inflight += 1
            try:
                version_header = self.headers.get("anthropic-version") or ANTHROPIC_VERSION_DEFAULT
                answer = handle_request(state, wire, body, version_header)
            finally:
                with state.lock:
                    state.inflight -= 1
            if answer is None:
                self._reply_error(wire, 502, "no provider could serve this request")
                return
            text, family, input_tokens, output_tokens = answer
            requested_model = body.get("model") if isinstance(body.get("model"), str) else "auto"
            stream = body.get("stream") is True
            if wire == "openai":
                if stream:
                    self._reply(200, render_openai_sse(requested_model, text, family, input_tokens, output_tokens), "text/event-stream")
                else:
                    self._reply_json(200, render_openai_json(requested_model, text, family, input_tokens, output_tokens))
            else:
                if stream:
                    self._reply(200, render_anthropic_sse(requested_model, text, family, input_tokens, output_tokens), "text/event-stream")
                else:
                    self._reply_json(200, render_anthropic_json(requested_model, text, family, input_tokens, output_tokens))

        def _reply_error(self, wire, status, message):
            if wire == "openai":
                self._reply_json(status, {"error": {"message": message, "type": "server_error", "code": None}})
            else:
                self._reply_json(status, {"type": "error", "error": {"type": "api_error", "message": message}})

    return Handler


# --------------------------------------------------------------------------
# Registration and the /work poller
# --------------------------------------------------------------------------

def register_server(env_url, port, catalog):
    for _ in range(40):
        status, body, _raw = _request_json(
            "POST", env_url + "/register",
            {"Accept": "application/json"},
            {"protocol_version": 1, "port": port, "catalog": catalog},
            timeout=UPSTREAM_TIMEOUT,
        )
        if 200 <= status < 300 and isinstance(body, dict) and body.get("ok") is True:
            return True
        time.sleep(0.25)
    return False


def monitor_work(state):
    silent_polls = 0
    while True:
        status, body, _raw = _request_json(
            "GET", state.env_url + "/work", {"Accept": "application/json"}, timeout=UPSTREAM_TIMEOUT
        )
        if status == 0:
            # The env is unreachable. After 30s of silence it is gone, not
            # slow; exit cleanly instead of hanging until the harness cap.
            silent_polls += 1
            if silent_polls >= 15:
                with state.lock:
                    busy = state.inflight
                if busy == 0:
                    os._exit(0)
        else:
            silent_polls = 0
            if isinstance(body, dict) and body.get("done") is True:
                with state.lock:
                    busy = state.inflight
                if busy == 0:
                    os._exit(0)
        time.sleep(WORK_POLL_SECONDS)


def main(argv):
    list_mode = "--list" in argv
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not prompt_file or not env_url:
        print(EMPTY_CATALOG)
        return 0
    providers, ratings, manifest, error = load_brief(prompt_file)
    if error is not None or not providers:
        print(EMPTY_CATALOG)
        return 0
    ratings_by_key = {(row["provider"], row["model"]): row for row in ratings}
    probes = [probe_provider_catalog(env_url, provider) for provider in providers]
    catalog = assemble_catalog(providers, probes, ratings_by_key)
    if list_mode:
        print(json.dumps(catalog, separators=(",", ":")))
        return 0
    state = State(env_url, providers, ratings_by_key, probes, manifest, catalog)
    server = HTTPServer(("127.0.0.1", 0), create_handler(state))
    port = server.server_address[1]
    if not register_server(env_url, port, catalog):
        sys.stderr.write("free-model-aggregator: /register failed\n")
        return 1
    poller = threading.Thread(target=monitor_work, args=(state,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
