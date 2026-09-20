from __future__ import annotations

import os
import re
import subprocess
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ToolKind = Literal["open_app", "open_url", "web_search", "list_files", "make_folder", "write_file"]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    message: str


@dataclass(frozen=True)
class ToolCall:
    kind: ToolKind
    args: dict
    requires_confirmation: bool = True

    def summarize(self) -> str:
        if self.kind == "open_app":
            return f"Open app: {self.args.get('app')}"
        if self.kind == "open_url":
            return f"Open URL: {self.args.get('url')}"
        if self.kind == "web_search":
            return f"Web search: {self.args.get('query')}"
        if self.kind == "list_files":
            return f"List files in: {self.args.get('path')}"
        if self.kind == "make_folder":
            return f"Create folder: {self.args.get('path')}"
        if self.kind == "write_file":
            return f"Write file: {self.args.get('path')}"
        return f"Tool: {self.kind}"


def _files_root() -> Path:
    # Restrict file actions to a dedicated safe directory.
    project_root = Path(__file__).resolve().parents[1]
    root = os.getenv("VOICE_AGENT_FILES_ROOT") or str(project_root / "agent-files")
    p = Path(root).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_under_root(user_path: str) -> Path:
    root = _files_root()
    # Treat relative paths as relative to the root.
    p = Path(user_path.strip())
    if not p.is_absolute():
        p = root / p
    p = p.expanduser().resolve()
    if root not in p.parents and p != root:
        raise PermissionError(f"Path must be inside {root}")
    return p


APP_ALIASES: dict[str, list[str]] = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "paint": ["mspaint.exe"],
    "explorer": ["explorer.exe"],
    "chrome": ["chrome.exe"],
    "edge": ["msedge.exe"],
}

BLOCKED_APPS: set[str] = {
    # Block shells/terminals to avoid turning "open app" into arbitrary command execution.
    "cmd",
    "cmd.exe",
    "powershell",
    "powershell.exe",
    "pwsh",
    "pwsh.exe",
    "wt",
    "wt.exe",
    "terminal",
    "windows terminal",
}


def parse_tool_call(text: str) -> ToolCall | None:
    t = (text or "").strip()
    if not t:
        return None

    tl = t.lower()

    # A) Open URL / web search
    m = re.search(r"\b(open|go to|visit)\s+(?P<url>https?://\S+)\b", t, flags=re.IGNORECASE)
    if m:
        return ToolCall(kind="open_url", args={"url": m.group("url").strip()}, requires_confirmation=True)

    m = re.search(r"\bsearch\s+(for\s+)?(?P<q>.+)$", t, flags=re.IGNORECASE)
    if m:
        q = m.group("q").strip().strip("\"'")
        if q:
            return ToolCall(kind="web_search", args={"query": q}, requires_confirmation=True)

    # B) Open apps
    m = re.search(r"\bopen\s+(?P<app>[a-zA-Z0-9 _-]+)\b", tl)
    if m:
        app = m.group("app").strip()
        app = re.sub(r"\s+", " ", app)
        # Normalize common phrases like "file explorer"
        app = app.replace("file explorer", "explorer").replace("windows explorer", "explorer")
        return ToolCall(kind="open_app", args={"app": app}, requires_confirmation=True)

    # D) Files (safe)
    m = re.search(r"\b(list|show)\s+(files|folder|directory)\s+(in\s+)?(?P<p>.+)$", t, flags=re.IGNORECASE)
    if m:
        p = m.group("p").strip().strip("\"'")
        return ToolCall(kind="list_files", args={"path": p or "."}, requires_confirmation=False)

    m = re.search(r"\b(create|make)\s+(a\s+)?(folder|directory)\s+(named\s+)?(?P<p>.+)$", t, flags=re.IGNORECASE)
    if m:
        p = m.group("p").strip().strip("\"'")
        return ToolCall(kind="make_folder", args={"path": p}, requires_confirmation=True)

    # "create file notes.txt with text hello"
    m = re.search(
        r"\b(create|make|write)\s+(a\s+)?file\s+(named\s+)?(?P<name>\S+)\s+(with\s+text|saying)\s+(?P<body>.+)$",
        t,
        flags=re.IGNORECASE,
    )
    if m:
        name = m.group("name").strip().strip("\"'")
        body = m.group("body").strip().strip("\"'")
        return ToolCall(kind="write_file", args={"path": name, "text": body}, requires_confirmation=True)

    return None


def execute_tool(call: ToolCall) -> ToolResult:
    try:
        if call.kind == "open_app":
            app = str(call.args.get("app") or "").strip().lower()
            if not app:
                return ToolResult(False, "No app specified.")
            if app in BLOCKED_APPS:
                return ToolResult(False, "That app is blocked for safety.")

            # Prefer known aliases, otherwise try opening by name/path.
            exe_candidates = APP_ALIASES.get(app)
            target = exe_candidates[0] if exe_candidates else str(call.args.get("app") or "").strip().strip("\"'")

            if sys.platform.startswith("win"):
                # Use PowerShell Start-Process to avoid cmd.exe parsing issues.
                subprocess.Popen(
                    [
                        "powershell",
                        "-NoProfile",
                        "-NonInteractive",
                        "-Command",
                        "Start-Process",
                        "-FilePath",
                        target,
                    ],
                    shell=False,
                )
            else:
                subprocess.Popen([target], shell=False)
            return ToolResult(True, f"Opened {target}.")

        if call.kind == "open_url":
            url = str(call.args.get("url") or "").strip()
            if not url.lower().startswith(("http://", "https://")):
                return ToolResult(False, "URL must start with http:// or https://")
            webbrowser.open(url)
            return ToolResult(True, "Opened the URL.")

        if call.kind == "web_search":
            q = str(call.args.get("query") or "").strip()
            if not q:
                return ToolResult(False, "No search query provided.")
            from urllib.parse import quote_plus
            url = "https://www.google.com/search?q=" + quote_plus(q)
            webbrowser.open(url)
            return ToolResult(True, f"Searching for {q}.")

        if call.kind == "list_files":
            p = _resolve_under_root(str(call.args.get("path") or "."))
            if not p.exists():
                return ToolResult(False, "That folder does not exist (inside the safe folder).")
            if p.is_file():
                return ToolResult(True, f"That is a file: {p.name}")
            items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            names = []
            for it in items[:30]:
                names.append((it.name + ("/" if it.is_dir() else "")))
            more = "" if len(items) <= 30 else f" (and {len(items) - 30} more)"
            root = _files_root()
            rel = str(p.relative_to(root)) if p != root else "."
            return ToolResult(True, f"Files in {rel}: " + ", ".join(names) + more)

        if call.kind == "make_folder":
            p = _resolve_under_root(str(call.args.get("path") or "").strip())
            p.mkdir(parents=True, exist_ok=True)
            root = _files_root()
            rel = str(p.relative_to(root)) if p != root else "."
            return ToolResult(True, f"Created folder {rel}.")

        if call.kind == "write_file":
            p = _resolve_under_root(str(call.args.get("path") or "").strip())
            text = str(call.args.get("text") or "")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
            root = _files_root()
            rel = str(p.relative_to(root)) if p != root else "."
            return ToolResult(True, f"Wrote {rel}.")

        return ToolResult(False, "Unknown tool.")
    except Exception as e:
        return ToolResult(False, f"Tool failed: {e}")
