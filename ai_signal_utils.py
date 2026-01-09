import json
import os
import re
import configparser
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Avoid noisy request logs if other libs enable them.
logging.getLogger("httpx").setLevel(logging.WARNING)


DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")


def _load_gemini_settings_from_config() -> Tuple[Optional[str], Optional[str]]:
    """Best-effort read of Gemini settings from config.ini.

    Precedence is handled by callers (env vars override config).
    """
    try:
        config = configparser.ConfigParser()
        config.read("config.ini")
        api_key = config.get("Gemini", "api_key", fallback=None)
        model = config.get("Gemini", "model", fallback=None)
        if isinstance(api_key, str):
            api_key = api_key.strip() or None
        if isinstance(model, str):
            model = model.strip() or None
        return api_key, model
    except Exception:
        return None, None


@dataclass(frozen=True)
class AiDecision:
    signal: bool
    confidence: float = 0.0
    reason: str = ""


# Symbol heuristic:
# - either exactly 6 letters (XAUUSD), OR
# - contains digits/underscore/dot (USTEC_X100m, BTCUSDm, US30.cash)
_SYMBOL_RE = re.compile(r"\b(?:[A-Z]{6}|[A-Z][A-Z0-9_.]{2,15}[0-9_.][A-Z0-9_.]*)\b", re.IGNORECASE)
_SIDE_RE = re.compile(r"\b(BUY|SELL)\b", re.IGNORECASE)


def _extract_symbol_candidate(text: str) -> Optional[str]:
    match = _SYMBOL_RE.search(text or "")
    if not match:
        return None
    return match.group(0).upper()


def fallback_should_forward(text: str) -> bool:
    if not text:
        return False
    if re.search(r"close half lots", text, re.IGNORECASE):
        return True
    # Keep this fallback conservative: only forward if it looks like a trade.
    return _extract_symbol_candidate(text) is not None and _SIDE_RE.search(text) is not None


def _get_genai_client(api_key: Optional[str]):
    try:
        from google import genai
    except Exception:
        return None

    try:
        # If api_key is None, the SDK will pick up GEMINI_API_KEY from env.
        return genai.Client(api_key=api_key) if api_key else genai.Client()
    except Exception:
        return None

def _json_only_prompt_classifier(message_text: str) -> str:
    return (
        "You are a strict professional classifier for Telegram trading signals. "
        "Return ONLY a single JSON object and nothing else (no markdown, no code fences, no explanation).\n\n"
        "Goal: decide whether the message is an actionable trade instruction (signal=true) or not (signal=false).\n\n"
        "Behavior rules (be conservative):\n"
        "- signal=true ONLY when the message contains a clear actionable trade instruction (explicit BUY or SELL for a tradable symbol and at least one risk/price field like SL, TP, entry or 'now' for market).\n"
        "- signal=false for greetings, chat, analysis, disclaimers, performance screenshots/results, news, or any ambiguous text.\n"
        "- Ignore noise words such as 'risky', 'signal', 'trade', channel names, and standard disclaimers.\n"
        "- When unsure, prefer signal=false.\n\n"
        "Output schema EXACTLY (keys must appear and types must match):\n"
        "{\"signal\": true|false, \"confidence\": 0.0, \"reason\": \"short explanation (one sentence)\"}\n\n"
        "Confidence guidance: 0.0 = definitely not, 1.0 = definitely yes. Use a decimal between 0 and 1.\n\n"
        f"Message:\n{message_text}"
    )



def _json_only_prompt_parser(message_text: str, default_lot: float, allowed_symbols: Optional[List[str]] = None) -> str:
    allowed_block = ""
    if allowed_symbols is not None:
        allowed_trimmed = [s for s in allowed_symbols if isinstance(s, str) and s.strip()][:200]
        if allowed_trimmed:
            allowed_block = (
                "ALLOWED_MT5_SYMBOLS (must match exactly one of these strings):\n"
                + " - "
                + "\n - ".join(allowed_trimmed)
                + "\n\n"
            )
        else:
            allowed_block = (
                "ALLOWED_MT5_SYMBOLS list is empty. If you cannot map the message symbol to an exact MT5 symbol, return signal=false.\n\n"
            )

    return (
        "You are a professional trading-signal parser. Extract structured trade instructions from a Telegram message. "
        "Return ONLY a single JSON object and nothing else (no markdown, no comments, no code fences).\n\n"
        "If the message is NOT a clear trade instruction, return signal=false (with confidence) and set other trade fields to null/empty as below.\n\n"
        "Output schema EXACTLY (keys must exist, types must match):\n"
        "{\n"
        "  \"signal\": true|false,\n"
        "  \"symbol\": \"XAUUSD\"|null,\n"
        "  \"side\": \"BUY\"|\"SELL\"|null,\n"
        "  \"order_kind\": \"MARKET\"|\"PENDING\"|null,\n"
        "  \"pending_type\": \"LIMIT\"|\"STOP\"|null,\n"
        "  \"entry\": 0.0|null,\n"
        "  \"sl\": 0.0|null,\n"
        "  \"tps\": [0.0],\n"
        "  \"lot\": 0.0|null,\n"
        "  \"confidence\": 0.0,\n"
        "  \"reason\": \"short explanation\"\n"
        "}\n\n"
        "Parsing notes (be strict and numeric):\n"
        "- Normalize side to BUY or SELL. If both appear, choose the primary instruction or return signal=false.\n"
        "- order_kind: MARKET when message implies immediate execution (contains 'now', 'market', or no entry price); PENDING when message specifies an entry price or 'limit'/'stop'.\n"
        "- pending_type: LIMIT when the message says 'LIMIT' or 'limit'; STOP when it says 'STOP' or 'stop'. Must be set if order_kind=PENDING.\n"
        "- entry: required numeric value when order_kind=PENDING; may be null for MARKET.\n"
        "- sl and each tp must be numeric (float). If multiple TP tokens are present, return them in the tps array in the order they appear.\n"
            f"- lot: numeric if provided in message; otherwise set to null (system default is {default_lot}).\n"
        "- confidence: 0.0..1.0, reflect how confident you are in the parsed fields.\n"
        "- reason: a short (<= 100 chars) justification for the decision (e.g., 'explicit BUY + SL + TP present').\n\n"
        "Symbol mapping rules:\n"
        "- Prefer exact MT5-style symbols such as XAUUSD, EURUSD, USTEC_X100m, US30.cash. Return the exact broker symbol string.\n"
        "- If ALLOWED_MT5_SYMBOLS is provided, the parsed symbol MUST exactly match one of those entries (including suffixes like '_X100m'); otherwise return signal=false.\n"
        "- Do NOT treat generic words (RISKY, TRADE, SIGNAL) as symbols.\n\n"
        "CRITICAL SYMBOL RULES (STRICT):\n"
        "- Abbreviations like UJ, GU, EU, EJ, GJ, NAS, GOLD are NOT valid MT5 symbols by themselves.\n"
        "        - You MAY map common abbreviations ONLY if the mapping is unambiguous:\n"
        "    UJ  -> USDJPY\n"
        "    GU  -> GBPUSD\n"
        "    EU  -> EURUSD\n"
        "    EJ  -> EURJPY\n"
        "    GJ  -> GBPJPY\n"   
        "    GOLD -> XAUUSD\n"
        "        - If multiple mappings are possible or broker symbol is unknown, return signal=false.\n"
        "        - NEVER output abbreviations like \"UJ\" as the final symbol.\n"
        "        - The final \"symbol\" MUST be a real MT5 broker symbol string.\n\n"
        "        - If the broker-specific symbol is unknown (not in ALLOWED_MT5_SYMBOLS), return signal=false.\n\n"
        "Noise handling:\n"
        "- Remove common disclaimers, channel names, repeated words, and punctuation noise before parsing. Words like 'risky', 'disclaimer', 'past profits' are noise.\n"
        "- Be robust to mixed case and extra whitespace.\n\n"
        "Conservative rule: if any required numeric field (entry when PENDING, sl when present in guidance, or at least one TP) is ambiguous or unparsable, return signal=false with a low confidence.\n\n"
        f"{allowed_block}"
        "Examples (DO NOT output these; they are guidance only):\n"
        "- 'BUY USTEC_X100m now\\nSL 25283\\nTP 25729' => signal=true, symbol=USTEC_X100m, side=BUY, order_kind=MARKET, sl=25283, tps=[25729]\n"
        "- 'XAUUSD SELL LIMIT 2043\\nSL 2051\\nTP 2035 TP 2027' => order_kind=PENDING, pending_type=LIMIT, entry=2043\n\n"
        "Message:\n"
        f"{message_text}"
    )


def _safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    # Strip code fences if the model returns them anyway.
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\n?", "", stripped)
        stripped = re.sub(r"\n?```$", "", stripped)
        stripped = stripped.strip()
    try:
        obj = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _validate_classifier(obj: Dict[str, Any]) -> Optional[AiDecision]:
    signal = obj.get("signal")
    if not isinstance(signal, bool):
        return None

    confidence = obj.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)):
        confidence = 0.0
    confidence = max(0.0, min(1.0, float(confidence)))

    reason = obj.get("reason", "")
    if not isinstance(reason, str):
        reason = ""

    return AiDecision(signal=signal, confidence=confidence, reason=reason)


def _validate_trade_intent(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Required keys must exist.
    required_keys = {
        "signal",
        "symbol",
        "side",
        "order_kind",
        "pending_type",
        "entry",
        "sl",
        "tps",
        "lot",
        "confidence",
        "reason",
    }
    if not required_keys.issubset(obj.keys()):
        return None

    if not isinstance(obj["signal"], bool):
        return None

    if not isinstance(obj["tps"], list):
        return None

    # Normalize common fields.
    normalized: Dict[str, Any] = dict(obj)

    if isinstance(normalized.get("symbol"), str):
        normalized["symbol"] = normalized["symbol"].strip().upper() or None

    if isinstance(normalized.get("side"), str):
        normalized["side"] = normalized["side"].strip().upper() or None

    if isinstance(normalized.get("order_kind"), str):
        normalized["order_kind"] = normalized["order_kind"].strip().upper() or None

    if isinstance(normalized.get("pending_type"), str):
        normalized["pending_type"] = normalized["pending_type"].strip().upper() or None

    # Coerce numerics where possible.
    for key in ["entry", "sl", "lot", "confidence"]:
        val = normalized.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)):
            normalized[key] = float(val)
            continue
        # Attempt string-to-float
        if isinstance(val, str):
            try:
                normalized[key] = float(val.strip())
            except Exception:
                normalized[key] = None

    # TP list coercion
    tps: List[float] = []
    for tp in normalized.get("tps") or []:
        if isinstance(tp, (int, float)):
            tps.append(float(tp))
        elif isinstance(tp, str):
            try:
                tps.append(float(tp.strip()))
            except Exception:
                continue
    normalized["tps"] = tps

    # Confidence clamp
    conf = normalized.get("confidence")
    if not isinstance(conf, (int, float)):
        normalized["confidence"] = 0.0
    else:
        normalized["confidence"] = max(0.0, min(1.0, float(conf)))

    reason = normalized.get("reason")
    if not isinstance(reason, str):
        normalized["reason"] = ""

    return normalized


def gemini_classify_sync(
    message_text: str,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 4.0,
) -> Optional[AiDecision]:
    cfg_api_key, cfg_model = _load_gemini_settings_from_config()
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or cfg_api_key
    model = (os.environ.get("GEMINI_MODEL") or model or cfg_model or DEFAULT_GEMINI_MODEL)
    if not api_key:
        return None

    client = _get_genai_client(api_key)
    if client is None:
        return None

    prompt = _json_only_prompt_classifier(message_text)

    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", None)
    except Exception as e:
        logging.warning("Gemini classify failed: %s: %s", type(e).__name__, str(e)[:200])
        return None

    obj = _safe_json_loads(text or "")
    if not obj:
        return None

    return _validate_classifier(obj)


async def gemini_classify_async(
    message_text: str,
    *,
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 4.0,
) -> Optional[AiDecision]:
    cfg_api_key, cfg_model = _load_gemini_settings_from_config()
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or cfg_api_key
    model = (os.environ.get("GEMINI_MODEL") or model or cfg_model or DEFAULT_GEMINI_MODEL)
    if not api_key:
        return None

    # The official SDK is sync; keep async API but run sync call.
    return gemini_classify_sync(
        message_text,
        api_key=api_key,
        model=model,
        timeout_seconds=timeout_seconds,
    )


def gemini_parse_trade_intent_sync(
    message_text: str,
    *,
    default_lot: float,
    allowed_symbols: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_GEMINI_MODEL,
    timeout_seconds: float = 8.0,
) -> Optional[Dict[str, Any]]:
    cfg_api_key, cfg_model = _load_gemini_settings_from_config()
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or cfg_api_key
    model = (os.environ.get("GEMINI_MODEL") or model or cfg_model or DEFAULT_GEMINI_MODEL)
    if not api_key:
        return None

    client = _get_genai_client(api_key)
    if client is None:
        return None

    prompt = _json_only_prompt_parser(
        message_text,
        default_lot=default_lot,
        allowed_symbols=allowed_symbols,
    )

    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = getattr(resp, "text", None)
    except Exception as e:
        logging.warning("Gemini parse failed: %s: %s", type(e).__name__, str(e)[:200])
        return None

    obj = _safe_json_loads(text or "")
    if not obj:
        return None

    normalized = _validate_trade_intent(obj)
    if not normalized:
        return None

    return normalized


def should_forward_message_sync(text: str) -> Tuple[bool, str]:
    """Decide whether to forward a message.

    Returns (should_forward, reason). Always fails back to the existing regex behavior.
    """
    if not text:
        return False, "empty"

    if re.search(r"close half lots", text, re.IGNORECASE):
        return True, "keyword:close half lots"

    # Cheap pre-gate: only consider messages containing a symbol candidate and side.
    symbol = _extract_symbol_candidate(text)
    if not symbol:
        return False, "no_symbol"
    if _SIDE_RE.search(text) is None:
        return False, "no_side"

    decision = gemini_classify_sync(text)
    if decision is None:
        return True, "fallback_regex"

    return decision.signal, f"ai:{decision.confidence:.2f}:{decision.reason}"[:200]


async def should_forward_message_async(text: str) -> Tuple[bool, str]:
    """Async version for Telethon forwarder."""
    if not text:
        return False, "empty"

    if re.search(r"close half lots", text, re.IGNORECASE):
        return True, "keyword:close half lots"

    symbol = _extract_symbol_candidate(text)
    if not symbol:
        return False, "no_symbol"
    if _SIDE_RE.search(text) is None:
        return False, "no_side"

    decision = await gemini_classify_async(text)
    if decision is None:
        return True, "fallback_regex"

    return decision.signal, f"ai:{decision.confidence:.2f}:{decision.reason}"[:200]
