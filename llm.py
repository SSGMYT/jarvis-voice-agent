from __future__ import annotations

import ast
import datetime as _dt
import operator as _op
import re as _re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMResponse:
    text: str


class Responder:
    def respond(self, user_text: str, *, history: list[dict[str, Any]] | None = None) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


def _safe_arithmetic(expr: str) -> float | None:
    """Evaluate a simple arithmetic expression without using eval()."""
    expr = (expr or "").strip()
    if not expr or not _re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr):
        return None

    ops = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
        ast.USub: _op.neg,
        ast.UAdd: _op.pos,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("unsupported expression")

    try:
        tree = ast.parse(expr, mode="eval")
        return _eval(tree)
    except Exception:
        return None


def _normalize_history(history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    """Keep only valid user/assistant turns for Groq chat history."""
    out: list[dict[str, str]] = []
    if not history:
        return out

    for item in history[-20:]:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] = f"{out[-1]['content']} {content}"
        else:
            out.append({"role": role, "content": content})

    if out and out[0]["role"] != "user":
        out.pop(0)
    return out


class FallbackResponder(Responder):
    def respond(self, user_text: str, *, history: list[dict[str, Any]] | None = None) -> LLMResponse:
        t = (user_text or "").strip()
        if not t:
            return LLMResponse(text="I didn't catch that. Please say it again.")

        t_l = t.lower()

        if any(p in t_l for p in ("what time", "time is it", "current time")):
            now = _dt.datetime.now().strftime("%I:%M %p").lstrip("0")
            return LLMResponse(text=f"It is {now}.")

        if any(p in t_l for p in ("what day", "what date", "today's date", "date is it")):
            today = _dt.date.today().strftime("%A, %B %d, %Y").replace(" 0", " ")
            return LLMResponse(text=f"Today is {today}.")

        expr = t_l
        expr = expr.replace("plus", "+").replace("minus", "-").replace("times", "*").replace("multiplied by", "*")
        expr = expr.replace("x", "*").replace("divided by", "/").replace("over", "/")
        expr = _re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", " ", expr)
        expr = _re.sub(r"\s+", " ", expr).strip()

        val = _safe_arithmetic(expr)
        if val is not None:
            if float(val).is_integer():
                val = int(val)
            return LLMResponse(text=f"The answer is {val}.")

        if any(p in t_l for p in ("hello", "hi", "hey")):
            return LLMResponse(text="Hi. What would you like to do?")

        return LLMResponse(
            text="I'm running offline without Groq, so I can only do basic tasks like time, date, and simple math."
        )


class GroqResponder(Responder):
    def __init__(self, *, api_key: str, model: str, system_prompt: str) -> None:
        from groq import Groq  # lazy import

        self.client = Groq(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt

    def respond(self, user_text: str, *, history: list[dict[str, Any]] | None = None) -> LLMResponse:
        user_text = (user_text or "").strip()
        if not user_text:
            return LLMResponse(text="I didn't catch that. Please say it again.")

        messages: list[dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(_normalize_history(history))
        messages.append({"role": "user", "content": user_text})

        try:
            chat = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.4,
                max_tokens=160,
            )
            choices = getattr(chat, "choices", None) or []
            if not choices:
                return LLMResponse(text="Groq returned no response. Please try again.")

            message = getattr(choices[0], "message", None)
            text = (getattr(message, "content", None) or "").strip()
            return LLMResponse(text=text or "Sorry, I'm not sure how to respond to that.")
        except Exception as e:
            msg = str(e).lower()
            if "invalid_api_key" in msg or "authentication" in msg or "401" in msg:
                return LLMResponse(
                    text="Groq rejected the API key. Please set a valid GROQ_API_KEY, then try again."
                )
            if "rate limit" in msg or "429" in msg:
                return LLMResponse(text="Groq rate limit reached. Please wait a moment and try again.")
            print(f"[Groq error] {e}")
            return LLMResponse(text="I had trouble reaching Groq. I'll keep running offline.")
