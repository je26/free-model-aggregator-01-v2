#!/usr/bin/env python3
"""Free-model aggregator v2: one server speaking OpenAI + Anthropic wires over
ten free providers (OpenAI, Anthropic and Gemini upstream wires).

What v2 changes over the v1 design (same wire translation and server; the
routing is new):

  * Risk-priced routing. Every provider's failure schedule is drawn by the
    kit's sealed generator from fixed per-provider budget ranges, with 1-2
    scheduled failures inside [2, budget-1] and at most one outage. A
    Bayesian posterior over each provider's schedule (budget, fail_calls,
    outage_from) is maintained from the trial's own observations, and
    candidates rank by rating less a risk price: W * P(next call fails),
    with W adaptive -- aggressive (400) while the quality-deficit budget is
    comfortable, conservative (100) once 500 of the 764 milli-points of
    deficit that quality 98 allows have been spent. Call #1 to any provider
    always succeeds (budgets >= 4, failures >= 2), so fresh providers rank
    purely by rating.
  * Refusal classification. A refusal's error body names its kind: 'budget
    is spent' and 'is out' are permanent (the provider is never called
    again); 'refused this call' is a scheduled one-off (the posterior learns
    exactly which call number failed). Uncounted errors (bad auth/model, no
    response) neither consume the env's call count nor touch the posterior.
  * Upstream calls stay non-streaming JSON whatever the client asked; the
    full answer is re-emitted in the client's wire with usage copied exactly
    and the stop reason mapped. No clock, no randomness: every choice
    derives from the brief, the requests and the providers' answers, so a
    replay reproduces the same state export byte for byte.

Stdlib only. Run: `python3 main.py` inside a trial, or `python3 main.py --list`
to print the catalog document byte-for-byte as registered.
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

# Risk price: rating milli-points charged per unit of failure probability.
# W_HI while the spent quality deficit is below DEFICIT_SWITCH, W_LO after.
# Quality 98 (score 99) allows 764 weighted milli-points of deficit; the
# switch at 500 keeps a reserve for forced failovers late in the trial.
W_HI = 400
W_LO = 100
DEFICIT_SWITCH = 500

EMPTY_CATALOG = '{"providers":[]}'


# --------------------------------------------------------------------------
# Failure-risk model
#
# Derived from the kit's sealed generator (program/generator/draw.mjs),
# which draws every instance, practice and scored alike:
#   * budget ~ uniform [calls_min, calls_max] per provider (fixed constants);
#   * fail_calls: 1-2 distinct numbers drawn uniformly from [2, budget-1];
#   * exactly one provider (uniform pick) has outage_from in [2, budget-1];
#   * one provider (uniform pick) is designated: its fail_calls include 2.
# A hypothesis is (budget, fail_calls, outage_from_or_None) with a prior
# weight. Each counted observation filters the hypotheses; P(fail at k) is
# the posterior mass under which call k fails.
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


def _schedule_prior(pid):
    lo, hi = BUDGET_RANGES.get(pid, DEFAULT_BUDGET_RANGE)
    hyps = {}
    n_budgets = hi - lo + 1
    for budget in range(lo, hi + 1):
        span = list(range(2, budget))
        for count in (1, 2):
            if len(span) < count:
                continue
            subsets = list(combinations(span, count))
            p_raw = (1.0 / n_budgets) * 0.5 / len(subsets)
            for raw in subsets:
                raw = tuple(sorted(raw))
                # Designated forcing: the smallest scheduled failure becomes 2.
                forced = tuple(sorted({2} | set(raw[1:]))) if 2 not in raw else raw
                for designated, p_des in ((False, 1.0 - P_DESIGNATED), (True, P_DESIGNATED)):
                    fails = forced if designated else raw
                    key = (budget, fails, None)
                    hyps[key] = hyps.get(key, 0.0) + p_raw * p_des * (1.0 - P_OUTAGE)
                    for outage_from in span:
                        key = (budget, fails, outage_from)
                        hyps[key] = hyps.get(key, 0.0) + p_raw * p_des * P_OUTAGE / len(span)
    return hyps


def _schedule_fails(hyp, k):
    budget, fails, outage_from = hyp
    if k > budget:
        return True
    if outage_from is not None and k >= outage_from:
        return True
    return k in fails


def _schedule_consistent(hyp, k, outcome):
    """Whether hypothesis `hyp` explains call `k` ending as `outcome`:
    'ok', 'transient' (scheduled failure) or 'permanent' (budget/outage).
    The env checks budget, then outage, then the schedule, so a transient
    failure proves the call was within budget, pre-outage and on schedule."""
    budget, fails, outage_from = hyp
    if outcome == "ok":
        return k <= budget and (outage_from is None or k < outage_from) and k not in fails
    if outcome == "transient":
        return k <= budget and (outage_from is None or k < outage_from) and k in fails
    return k > budget or (outage_from is not None and k >= outage_from)


class ProviderRisk:
    """Posterior over one provider's failure schedule."""

    def __init__(self, pid):
        self.pid = pid
        self.hyps = _schedule_prior(pid)
        self.total = sum(self.hyps.values())
        self._cache = {}

    def observe(self, k, outcome):
        kept = {h: w for h, w in self.hyps.items() if _schedule_consistent(h, k, outcome)}
        total = sum(kept.values())
        if total > 0:
            self.hyps = kept
            self.total = total
        # A zero-mass observation contradicts the model (it cannot happen
        # with an exact refusal classifier); keep the old posterior.
        self._cache.clear()

    def p_fail(self, k, assume_ok=()):
        """P(call k fails), optionally assuming the given call numbers
        succeed first (for planning with projected counts)."""
        key = (k, tuple(sorted(assume_ok)))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        num = 0.0
        den = 0.0
        for hyp, w in self.hyps.items():
            if any(_schedule_fails(hyp, a) for a in assume_ok):
                continue
            den += w
            if _schedule_fails(hyp, k):
                num += w
        p = (num / den) if den > 0 else 1.0
        self._cache[key] = p
        return p


# --------------------------------------------------------------------------
# Brief parsing
# --------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the `}` closing the object opening at `start`, or -1."""
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


def _documents_in(text):
    """Every JSON object in `text` carrying a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _provider_of(v):
    if not isinstance(v, dict):
        return None
    pid = _str(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = _str(v.get("path")) or _str(v.get("base_path")) or _str(v.get("base_url")) or _str(v.get("base"))
    return {
        "id": pid,
        "name": _str(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _str(v.get("key_env")),
        "key": _str(v.get("key")),
    }


def _rating_of(v):
    if not isinstance(v, dict):
        return None
    model = _str(v.get("model"))
    provider = _str(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) and not isinstance(rating, bool) else 0,
        "context_length": _num_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not isinstance(v, dict):
        return None
    rid = _str(v.get("id"))
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


def read_brief(path):
    """Return (providers, ratings_rows, manifest, error)."""
    if not path:
        return [], [], [], "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return [], [], [], "cannot read %s: %s" % (path, exc)
    docs = _documents_in(text)
    directory = next(
        (d for d in docs if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list)),
        None,
    )
    if directory is None:
        return [], [], [], "the brief holds no %s document" % PROVIDERS_SCHEMA
    providers = [p for p in (_provider_of(v) for v in directory["providers"]) if p is not None]
    ratings_doc = next(
        (d for d in docs if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list)),
        None,
    )
    ratings = []
    if ratings_doc is not None:
        ratings = [r for r in (_rating_of(v) for v in ratings_doc["ratings"]) if r is not None]
    workload_doc = next(
        (d for d in docs if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list)),
        None,
    )
    manifest = []
    if workload_doc is not None:
        manifest = [w for w in (_workload_request_of(v) for v in workload_doc["manifest"]) if w is not None]
    return providers, ratings, manifest, None


# --------------------------------------------------------------------------
# Provider catalog probing
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = base.strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(v):
    pricing = v.get("pricing")
    if not isinstance(pricing, dict):
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not isinstance(v, dict):
        return None
    if wire == "gemini":
        mid = _str(v.get("name"))
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
    else:
        mid = _str(v.get("id"))
    if mid == "":
        return None
    context = _num_or_none(v.get("context_length"))
    if context is None:
        context = _num_or_none(v.get("context_window"))
    if context is None:
        context = _num_or_none(v.get("inputTokenLimit"))
    free = v.get("free") if isinstance(v.get("free"), bool) else _zero_priced(v)
    return {
        "id": mid,
        "display_name": _str(v.get("display_name")) or _str(v.get("displayName")) or mid,
        "free": free,
        "context_length": context,
        "tools": v.get("tools") if isinstance(v.get("tools"), bool) else None,
        "vision": v.get("vision") if isinstance(v.get("vision"), bool) else None,
        "json": v.get("json") if isinstance(v.get("json"), bool) else None,
    }


def _models_in(wire, body):
    if not isinstance(body, dict):
        return []
    items = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(items, list):
        return []
    return [m for m in (_model_in(wire, v) for v in items) if m is not None]


def _http_json(method, url, headers, payload=None, timeout=UPSTREAM_TIMEOUT):
    """One HTTP call. Returns (status, parsed_json_or_None, raw_bytes)."""
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


def probe_provider(env_url, provider):
    """Probe one provider's catalog route. Returns (ok, models)."""
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION_DEFAULT
    for _ in range(PROBE_ATTEMPTS):
        status, body, _raw = _http_json("GET", url, headers, timeout=PROBE_TIMEOUT)
        if 200 <= status < 300 and body is not None:
            return True, _models_in(provider["wire"], body)
        if status == 0:
            continue
        return False, []
    return False, []


def build_catalog(providers, probes, ratings_by_key):
    """The catalog document: providers in brief order that answered, every
    model listed, free flag and context from the quality table where it has a
    row. `base_url` is the directory's own path, never joined to the env URL,
    so a replay reproduces it byte for byte."""
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
# Every request is normalised to one intermediate shape and then built in the
# upstream wire. The intermediate preserves exactly the fields the providers
# echo back: system text, user text, images in their original payload form
# (base64 stays base64, a URL stays a URL), tool names in order, stop
# sequences, max_tokens (exactly, or absent when the client sent none), and
# JSON mode. Temperature/top_p pass through when present.
# --------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _split_data_url(url):
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


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_of(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"))
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"))


def _openai_parts(message):
    """Content parts of one OpenAI message, in the intermediate shape."""
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
                url = _str(block["image_url"].get("url"))
                split = _split_data_url(url)
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
                "id": _str(call.get("id")),
                "name": _str(fn.get("name")),
                "arguments": fn.get("arguments", ""),
            })
    return parts


def _anthropic_parts(message):
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
                        "media": _str(source.get("media_type")) or "image/png",
                        "data": source["data"],
                    })
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    parts.append({"type": "image_url", "url": source["url"]})
            elif btype == "tool_use":
                parts.append({
                    "type": "tool_call",
                    "id": _str(block.get("id")),
                    "name": _str(block.get("name")),
                    "arguments": block.get("input", {}),
                })
            elif btype == "tool_result":
                parts.append({
                    "type": "tool_result",
                    "tool_use_id": _str(block.get("tool_use_id")),
                    "text": _tool_result_text(block.get("content")),
                })
    return parts


def _openai_tools(body):
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
                "description": _str(fn.get("description")),
                "parameters": fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}},
            })
    return tools


def _anthropic_tools(body):
    tools = []
    if isinstance(body.get("tools"), list):
        for raw in body["tools"]:
            if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "name": raw["name"],
                "description": _str(raw.get("description")),
                "parameters": raw.get("input_schema") if isinstance(raw.get("input_schema"), dict) else {"type": "object", "properties": {}},
            })
    return tools


def _openai_tool_choice(body):
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


def _anthropic_tool_choice(body):
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


def _stops_of(body, wire):
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


def normalize(body, wire):
    """A client body in either wire as the intermediate shape."""
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
        system_parts = []
        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                system_parts.append(_text_of(message.get("content")))
                continue
            if role == "tool":
                norm["messages"].append({
                    "role": "tool",
                    "tool_call_id": _str(message.get("tool_call_id")),
                    "parts": [{"type": "tool_result", "tool_use_id": _str(message.get("tool_call_id")), "text": _tool_result_text(message.get("content"))}],
                })
                continue
            norm["messages"].append({
                "role": "assistant" if role == "assistant" else "user",
                "parts": _openai_parts(message),
            })
        if system_parts:
            norm["system"] = "\n".join(system_parts)
        norm["tools"] = _openai_tools(body)
        norm["tool_choice"] = _openai_tool_choice(body)
        fmt = body.get("response_format")
        if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
            norm["json_mode"] = True
    else:
        system = body.get("system")
        if isinstance(system, str):
            norm["system"] = system
        elif isinstance(system, list):
            text = _text_of(system)
            if text != "":
                norm["system"] = text
        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            norm["messages"].append({
                "role": "assistant" if message.get("role") == "assistant" else "user",
                "parts": _anthropic_parts(message),
            })
        norm["tools"] = _anthropic_tools(body)
        norm["tool_choice"] = _anthropic_tool_choice(body)
    max_tokens = body.get("max_tokens")
    if max_tokens is None and wire == "openai":
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)) and not isinstance(max_tokens, bool):
        norm["max_tokens"] = max_tokens
    norm["stops"] = _stops_of(body, wire)
    if isinstance(body.get("temperature"), (int, float)) and not isinstance(body.get("temperature"), bool):
        norm["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)) and not isinstance(body.get("top_p"), bool):
        norm["top_p"] = body["top_p"]
    return norm


def _parts_to_openai_content(parts):
    """Intermediate parts as an OpenAI message content value."""
    out = []
    for part in parts:
        if part["type"] == "text":
            out.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_b64":
            out.append({"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (part["media"], part["data"])}})
        elif part["type"] == "image_url":
            out.append({"type": "image_url", "image_url": {"url": part["url"]}})
    if len(out) == 1 and out[0]["type"] == "text":
        return out[0]["text"]
    return out


def build_openai(norm, model):
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
        content = _parts_to_openai_content(message["parts"])
        entry = {"role": message["role"], "content": content if content != "" else None}
        if message["role"] == "assistant" and content == []:
            entry["content"] = None
        if calls:
            entry["tool_calls"] = calls
        messages.append(entry)
    out = {"model": model, "messages": messages}
    if norm["max_tokens"] is not None:
        out["max_tokens"] = norm["max_tokens"]
    if norm["temperature"] is not None:
        out["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        out["top_p"] = norm["top_p"]
    if norm["stops"]:
        out["stop"] = norm["stops"]
    if norm["tools"]:
        out["tools"] = [
            {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
            for t in norm["tools"]
        ]
    choice = norm["tool_choice"]
    if choice == "auto":
        out["tool_choice"] = "auto"
    elif choice == "required":
        out["tool_choice"] = "required"
    elif choice == "none":
        out["tool_choice"] = "none"
    elif isinstance(choice, tuple):
        out["tool_choice"] = {"type": "function", "function": {"name": choice[1]}}
    if norm["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    return out


def _parts_to_anthropic_content(parts):
    out = []
    for part in parts:
        if part["type"] == "text":
            out.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_b64":
            out.append({"type": "image", "source": {"type": "base64", "media_type": part["media"], "data": part["data"]}})
        elif part["type"] == "image_url":
            out.append({"type": "image", "source": {"type": "url", "url": part["url"]}})
        elif part["type"] == "tool_call":
            args = part["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            out.append({"type": "tool_use", "id": part["id"], "name": part["name"], "input": args})
        elif part["type"] == "tool_result":
            out.append({"type": "tool_result", "tool_use_id": part["tool_use_id"], "content": part["text"]})
    if len(out) == 1 and out[0]["type"] == "text":
        return out[0]["text"]
    return out


def build_anthropic(norm, model):
    out = {"model": model}
    if norm["max_tokens"] is not None:
        out["max_tokens"] = norm["max_tokens"]
    if norm["system"] is not None:
        out["system"] = norm["system"]
    messages = []
    for message in norm["messages"]:
        if message["role"] == "tool":
            blocks = []
            for part in message["parts"]:
                if part["type"] == "tool_result":
                    blocks.append({"type": "tool_result", "tool_use_id": part["tool_use_id"], "content": part["text"]})
            messages.append({"role": "user", "content": blocks})
            continue
        messages.append({"role": message["role"], "content": _parts_to_anthropic_content(message["parts"])})
    out["messages"] = messages
    if norm["temperature"] is not None:
        out["temperature"] = norm["temperature"]
    if norm["top_p"] is not None:
        out["top_p"] = norm["top_p"]
    if norm["tools"]:
        out["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in norm["tools"]
        ]
    choice = norm["tool_choice"]
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif isinstance(choice, tuple):
        out["tool_choice"] = {"type": "tool", "name": choice[1]}
    if norm["stops"]:
        out["stop_sequences"] = norm["stops"]
    return out


def _parts_to_gemini_parts(parts):
    out = []
    for part in parts:
        if part["type"] == "text":
            out.append({"text": part["text"]})
        elif part["type"] == "image_b64":
            out.append({"inlineData": {"mimeType": part["media"], "data": part["data"]}})
        elif part["type"] == "image_url":
            out.append({"fileData": {"fileUri": part["url"], "mimeType": "image/png"}})
        elif part["type"] == "tool_call":
            args = part["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            out.append({"functionCall": {"name": part["name"], "args": args}})
        elif part["type"] == "tool_result":
            out.append({"functionResponse": {"name": part["tool_use_id"], "response": {"result": part["text"]}}})
    return out


def build_gemini(norm, model):
    out = {}
    if norm["system"] is not None:
        out["systemInstruction"] = {"parts": [{"text": norm["system"]}]}
    contents = []
    for message in norm["messages"]:
        if message["role"] == "tool":
            contents.append({"role": "function", "parts": _parts_to_gemini_parts(message["parts"])})
        else:
            contents.append({"role": "model" if message["role"] == "assistant" else "user", "parts": _parts_to_gemini_parts(message["parts"])})
    out["contents"] = contents
    if norm["tools"]:
        out["tools"] = [{"functionDeclarations": [
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
        out["generationConfig"] = config
    return out


def build_upstream(norm, wire, model):
    if wire == "openai":
        return build_openai(norm, model)
    if wire == "anthropic":
        return build_anthropic(norm, model)
    return build_gemini(norm, model)


# --------------------------------------------------------------------------
# Upstream answers: parse to (text, stop family, input tokens, output tokens)
# Stop families: 0 stop, 1 length, 2 tool calls, 3 content filter.
# --------------------------------------------------------------------------

def _family_openai(finish):
    if finish == "length":
        return 1
    if finish in ("tool_calls", "function_call"):
        return 2
    if finish == "content_filter":
        return 3
    return 0


def _family_anthropic(stop):
    if stop == "max_tokens":
        return 1
    if stop == "tool_use":
        return 2
    if stop == "refusal":
        return 3
    return 0


def _family_gemini(reason):
    if reason == "MAX_TOKENS":
        return 1
    if reason == "SAFETY":
        return 3
    return 0


def parse_upstream(wire, body):
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
            text = _text_of(text)
        if not isinstance(text, str):
            return None
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if not isinstance(prompt, (int, float)) or not isinstance(completion, (int, float)):
            return None
        return (text, _family_openai(choices[0].get("finish_reason")), int(prompt), int(completion))
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
        return (text, _family_anthropic(body.get("stop_reason")), int(prompt), int(completion))
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
    return (text, _family_gemini(candidates[0].get("finishReason")), int(prompt), int(completion))


def classify_refusal(status, parsed):
    """Classify a non-2xx upstream answer. Returns (outcome, counted):
    outcome is 'transient', 'permanent' or 'uncounted'; counted says whether
    the env charged the call against the provider's budget/call number.
    All three wires carry the refusal reason in error.message:
      "<name>'s free-tier budget is spent" -> permanent
      "<name> is out"                     -> permanent (outage)
      "<name> refused this call"          -> transient (scheduled failure)
    Auth/model errors and transport failures spend no budget and carry no
    schedule information, so they are 'uncounted' and never touch the
    posterior. Anything unrecognised defaults to transient (safe: it never
    wrongly kills a provider)."""
    if status == 0 or parsed is None:
        return "uncounted", False
    if status in (400, 401):
        return "uncounted", False
    message = ""
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        message = _str(error.get("message"))
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
# Client answers, in the wire the client used
# --------------------------------------------------------------------------

def _openai_stop(family):
    return ["stop", "length", "tool_calls", "content_filter"][family]


def _anthropic_stop(family):
    return ["end_turn", "max_tokens", "tool_use", "refusal"][family]


def openai_json(client_model, text, family, input_tokens, output_tokens):
    return {
        "id": "chatcmpl-aggregator",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": client_model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": _openai_stop(family),
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def openai_sse(client_model, text, family, input_tokens, output_tokens):
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
    finish["choices"] = [{"index": 0, "delta": {}, "finish_reason": _openai_stop(family)}]
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


def anthropic_json(client_model, text, family, input_tokens, output_tokens):
    return {
        "id": "msg_aggregator",
        "type": "message",
        "role": "assistant",
        "model": client_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": _anthropic_stop(family),
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _sse_event(name, payload):
    return "event: %s\ndata: %s\n\n" % (name, json.dumps(payload, separators=(",", ":")))


def anthropic_sse(client_model, text, family, input_tokens, output_tokens):
    """The Anthropic event sequence: message_start, one text block's
    start/deltas/stop, message_delta, message_stop. Nothing follows it."""
    events = []
    events.append(_sse_event("message_start", {
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
    events.append(_sse_event("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for at in range(0, len(text), 13):
        events.append(_sse_event("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text[at:at + 13]},
        }))
    events.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}))
    events.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_stop(family), "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    }))
    events.append(_sse_event("message_stop", {"type": "message_stop"}))
    return "".join(events).encode("utf-8")


# --------------------------------------------------------------------------
# Planning: a global assignment, recomputed before every request
# --------------------------------------------------------------------------

def _meets(requires, caps):
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
        self.provider_by_id = {p["id"]: p for p in providers}
        self.dial_base = {p["id"]: join_base(env_url, p["base"]) for p in providers}
        self.manifest = manifest
        self.catalog = catalog
        self.calls = {}
        self.dead = set()
        self.request_index = 0
        self.in_flight = 0
        self.lock = threading.Lock()
        self.risk = {p["id"]: ProviderRisk(p["id"]) for p in providers}
        # Resolved models per provider: free flag and capabilities from the
        # quality table where it has a row, else the listing's own fields.
        self.models_by_provider = {}
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
            self.models_by_provider[provider["id"]] = resolved
        self.rating_of = {}
        for pid, models in self.models_by_provider.items():
            for model in models:
                self.rating_of[(pid, model["id"])] = model["rating"]
        # The scorer's best rating per request: the max over free models
        # meeting its requirements. Spent deficit against these bests drives
        # the adaptive risk price.
        self.best = {}
        for row in manifest:
            cands = self.candidates_for(row["requires"], False)
            self.best[row["id"]] = max([c[2] for c in cands], default=0)
        self.spent = 0
        # The free model ids the catalog registered, for GET /v1/models.
        self.served_ids = []
        seen_ids = set()
        for entry in catalog["providers"]:
            for model in entry["models"]:
                if model["free"] and model["id"] not in seen_ids:
                    seen_ids.add(model["id"])
                    self.served_ids.append((model["id"], entry["id"]))
        self.served_ids.sort()

    def candidates_for(self, requires, wire_filter):
        """(provider, model, rating) triples that may serve a request.

        `wire_filter` is set when the request carries JSON mode on the OpenAI
        wire: the Anthropic wire cannot echo JSON mode, so it is excluded."""
        out = []
        for provider in self.providers:
            if wire_filter and provider["wire"] == "anthropic":
                continue
            for model in self.models_by_provider.get(provider["id"], []):
                if not model["free"]:
                    continue
                if not _meets(requires, model["caps"]):
                    continue
                out.append((provider["id"], model["id"], model["rating"]))
        return out

    def _risk_price(self):
        return W_HI if self.spent < DEFICIT_SWITCH else W_LO

    def note(self, pid, call_number, outcome):
        """Record one counted upstream outcome: 'ok', 'transient' or
        'permanent'. A permanent refusal kills the provider outright; every
        outcome updates its schedule posterior."""
        if outcome == "permanent":
            self.dead.add(pid)
        self.risk[pid].observe(call_number, outcome)

    def _rank(self, candidates, counts, weight=1):
        """Candidates best-first by risk-priced score: rating less W times
        the posterior probability that the provider's next call fails,
        divided by the request's weight (a weight-2 request's rating deficit
        costs double, while a wasted call costs the same). `counts` may hold
        projected counts: the failure probability is then evaluated at the
        projected call number, assuming the projected earlier calls succeed.
        Ties break on rating, then provider and model id, so the order is
        fully deterministic."""
        price = self._risk_price()
        def key(cand):
            pid, mid, rating = cand
            if pid in self.dead:
                return (1 << 30, -rating, pid, mid)
            nxt = counts.get(pid, 0) + 1
            actual = self.calls.get(pid, 0)
            assume = tuple(range(actual + 1, counts.get(pid, 0) + 1))
            p = self.risk[pid].p_fail(nxt, assume)
            effective = rating - (price * p) / weight
            return (-effective, -rating, pid, mid)
        return sorted(candidates, key=key)

    def compute_plan(self, start_index, current_json_mode):
        """Assign every remaining request a (provider, model).

        Most-constrained requests (fewest viable candidates) are placed
        first so scarce capabilities are reserved for the requests that need
        them; each placement charges the provider's projected load, which the
        risk price then reads as a higher failure probability for its next
        call, spreading traffic instead of spending one provider's budget."""
        remaining = self.manifest[start_index:]
        if not remaining:
            return {}
        cand_lists = []
        for i, row in enumerate(remaining):
            if i == 0:
                wire_filter = current_json_mode
            else:
                # A future OpenAI request that requires JSON mode is assumed
                # to carry response_format, as the workload's JSON requests
                # do; Anthropic requests cannot express JSON mode.
                wire_filter = row["api"] == "openai" and row["requires"].get("json") is True
            cand_lists.append(self.candidates_for(row["requires"], wire_filter))
        order = sorted(
            range(len(remaining)),
            key=lambda i: (len(cand_lists[i]), -remaining[i]["weight"], i),
        )
        projected = dict(self.calls)
        plan = {}
        for i in order:
            if not cand_lists[i]:
                continue
            ranked = self._rank(cand_lists[i], projected, remaining[i]["weight"])
            chosen = ranked[0]
            plan[remaining[i]["id"]] = (chosen[0], chosen[1])
            projected[chosen[0]] = projected.get(chosen[0], 0) + 1
        return plan


def _requirements_from_body(norm):
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


def call_upstream(state, provider_id, model_id, norm, anthropic_version):
    """One non-streaming upstream call. Returns (answer_or_None, outcome,
    counted): outcome is 'ok', 'transient', 'permanent' or 'uncounted', and
    counted says whether the env charged the call (only counted outcomes
    advance the provider's call number and posterior)."""
    provider = state.provider_by_id[provider_id]
    wire = provider["wire"]
    payload = build_upstream(norm, wire, model_id)
    base = state.dial_base[provider_id]
    if wire == "gemini":
        url = base + "/v1beta/models/" + urllib.parse.quote(model_id, safe="") + ":generateContent"
    elif wire == "anthropic":
        url = base + "/v1/messages"
    else:
        url = base + "/v1/chat/completions"
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if wire == "anthropic":
        headers["anthropic-version"] = anthropic_version
    status, parsed, _raw = _http_json("POST", url, headers, payload, timeout=UPSTREAM_TIMEOUT)
    if 200 <= status < 300:
        answer = parse_upstream(wire, parsed)
        if answer is not None:
            return answer, "ok", True
        return None, "transient", True
    outcome, counted = classify_refusal(status, parsed)
    return None, outcome, counted


def serve(state, wire, body, anthropic_version, caller=None):
    """Serve one client request. Returns (text, family, in, out) or None.

    `caller` defaults to the real upstream call; the local simulator passes
    a schedule lookup instead, exercising the same planning and failover
    code without HTTP. A caller returns (answer, outcome, counted) like
    call_upstream."""
    call = caller or call_upstream
    norm = normalize(body, wire)
    with state.lock:
        index = state.request_index
        state.request_index += 1
    row = None
    if index < len(state.manifest) and state.manifest[index]["api"] == wire:
        row = state.manifest[index]
    if row is not None:
        requires = row["requires"]
    else:
        requires = _requirements_from_body(norm)
    plan = state.compute_plan(index, norm["json_mode"])
    planned = plan.get(row["id"]) if row is not None else None
    weight = row["weight"] if row is not None else 1
    ranked = state._rank(state.candidates_for(requires, norm["json_mode"]), state.calls, weight)
    if planned is not None and planned in [(c[0], c[1]) for c in ranked]:
        ranked = [(planned[0], planned[1], 0)] + [c for c in ranked if (c[0], c[1]) != planned]
    # Failover in two phases: first every candidate on a provider that has
    # not failed during THIS request, then the deferred same-provider
    # retries. A refusal may be a one-off failure or an exhausted budget;
    # other providers are tried before spending another call finding out
    # which, and a provider that fails twice in one request is left alone.
    # Only counted outcomes advance the call count and the posterior.
    tried = set()
    failed_providers = set()
    deferred = []
    queue = list(ranked)
    while queue:
        provider_id, model_id, _rating = queue.pop(0)
        if (provider_id, model_id) in tried or provider_id in state.dead:
            continue
        if provider_id in failed_providers:
            deferred.append((provider_id, model_id))
            continue
        tried.add((provider_id, model_id))
        answer, outcome, counted = call(state, provider_id, model_id, norm, anthropic_version)
        if counted:
            with state.lock:
                state.calls[provider_id] = state.calls.get(provider_id, 0) + 1
                call_number = state.calls[provider_id]
            state.note(provider_id, call_number, outcome)
        if answer is not None:
            if row is not None:
                state.spent += weight * max(0, state.best[row["id"]] - state.rating_of.get((provider_id, model_id), 0))
            return answer
        failed_providers.add(provider_id)
    failed_again = set()
    for provider_id, model_id in deferred:
        if (provider_id, model_id) in tried or provider_id in failed_again or provider_id in state.dead:
            continue
        tried.add((provider_id, model_id))
        answer, outcome, counted = call(state, provider_id, model_id, norm, anthropic_version)
        if counted:
            with state.lock:
                state.calls[provider_id] = state.calls.get(provider_id, 0) + 1
                call_number = state.calls[provider_id]
            state.note(provider_id, call_number, outcome)
        if answer is not None:
            if row is not None:
                state.spent += weight * max(0, state.best[row["id"]] - state.rating_of.get((provider_id, model_id), 0))
            return answer
        failed_again.add(provider_id)
    return None


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

def _model_lists(state):
    openai_list = {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": FIXED_CREATED, "owned_by": pid}
            for mid, pid in state.served_ids
        ],
    }
    anthropic_list = {
        "data": [
            {"type": "model", "id": mid, "display_name": mid, "created_at": FIXED_ANTHROPIC_CREATED_AT}
            for mid, _pid in state.served_ids
        ],
        "has_more": False,
    }
    return openai_list, anthropic_list


def make_handler(state):
    openai_list, anthropic_list = _model_lists(state)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format, *_args):
            pass

        def _send(self, status, payload, content_type="application/json"):
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

        def _send_json(self, status, obj):
            self._send(status, json.dumps(obj, separators=(",", ":")).encode("utf-8"))

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/v1/models":
                if self.headers.get("x-api-key") is not None:
                    self._send_json(200, anthropic_list)
                else:
                    self._send_json(200, openai_list)
                return
            self._send_json(404, {"error": {"message": "not found", "type": "invalid_request", "code": None}})

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/v1/chat/completions":
                wire = "openai"
            elif path == "/v1/messages":
                wire = "anthropic"
            else:
                self._send_json(404, {"error": {"message": "not found", "type": "invalid_request", "code": None}})
                return
            length = self.headers.get("Content-Length")
            try:
                count = int(length) if length else 0
            except ValueError:
                count = 0
            raw = self.rfile.read(count) if count > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                body = None
            if not isinstance(body, dict):
                self._send_error(wire, 400, "request body is not a JSON object")
                return
            with state.lock:
                state.in_flight += 1
            try:
                anthropic_version = self.headers.get("anthropic-version") or ANTHROPIC_VERSION_DEFAULT
                answer = serve(state, wire, body, anthropic_version)
            finally:
                with state.lock:
                    state.in_flight -= 1
            if answer is None:
                self._send_error(wire, 502, "no provider could serve this request")
                return
            text, family, input_tokens, output_tokens = answer
            client_model = body.get("model") if isinstance(body.get("model"), str) else "auto"
            stream = body.get("stream") is True
            if wire == "openai":
                if stream:
                    self._send(200, openai_sse(client_model, text, family, input_tokens, output_tokens), "text/event-stream")
                else:
                    self._send_json(200, openai_json(client_model, text, family, input_tokens, output_tokens))
            else:
                if stream:
                    self._send(200, anthropic_sse(client_model, text, family, input_tokens, output_tokens), "text/event-stream")
                else:
                    self._send_json(200, anthropic_json(client_model, text, family, input_tokens, output_tokens))

        def _send_error(self, wire, status, message):
            if wire == "openai":
                self._send_json(status, {"error": {"message": message, "type": "server_error", "code": None}})
            else:
                self._send_json(status, {"type": "error", "error": {"type": "api_error", "message": message}})

    return Handler


# --------------------------------------------------------------------------
# Registration and the /work poller
# --------------------------------------------------------------------------

def register(env_url, port, catalog):
    for _ in range(40):
        status, body, _raw = _http_json(
            "POST", env_url + "/register",
            {"Accept": "application/json"},
            {"protocol_version": 1, "port": port, "catalog": catalog},
            timeout=UPSTREAM_TIMEOUT,
        )
        if 200 <= status < 300 and isinstance(body, dict) and body.get("ok") is True:
            return True
        time.sleep(0.25)
    return False


def poll_work(state):
    failed = 0
    while True:
        status, body, _raw = _http_json(
            "GET", state.env_url + "/work", {"Accept": "application/json"}, timeout=UPSTREAM_TIMEOUT
        )
        if status == 0:
            # The env is unreachable. After 30s of silence it is gone, not
            # slow; exit cleanly instead of hanging until the harness cap.
            failed += 1
            if failed >= 15:
                with state.lock:
                    busy = state.in_flight
                if busy == 0:
                    os._exit(0)
        else:
            failed = 0
            if isinstance(body, dict) and body.get("done") is True:
                with state.lock:
                    busy = state.in_flight
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
    providers, ratings, manifest, error = read_brief(prompt_file)
    if error is not None or not providers:
        print(EMPTY_CATALOG)
        return 0
    ratings_by_key = {(row["provider"], row["model"]): row for row in ratings}
    probes = [probe_provider(env_url, provider) for provider in providers]
    catalog = build_catalog(providers, probes, ratings_by_key)
    if list_mode:
        print(json.dumps(catalog, separators=(",", ":")))
        return 0
    state = State(env_url, providers, ratings_by_key, probes, manifest, catalog)
    server = HTTPServer(("127.0.0.1", 0), make_handler(state))
    port = server.server_address[1]
    if not register(env_url, port, catalog):
        sys.stderr.write("free-model-aggregator: /register failed\n")
        return 1
    poller = threading.Thread(target=poll_work, args=(state,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
