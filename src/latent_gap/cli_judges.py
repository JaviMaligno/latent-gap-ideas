"""Subscription models (judges and generators) through the Claude Code and Codex CLIs (no API spend), isolated as in agent-telephone's
backends: allowlisted env, no user CLAUDE.md / AGENTS.md / MCP / tools / memory, fresh temp dir per call, stdin
closed. Each call is a single stateless prompt that must answer with JSON matching `schema`."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

ENV_ALLOW = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "SHELL")
CLAUDE_ISOLATION = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
CODEX_HOME = Path.home() / ".local/state/latent-gap-ideas/codex-home"
WAITS = (30, 120, 600, 1800, 3600, 3600)  # usage limits and network drops are waited out, then the call fails
TRANSIENT = ("ENOTFOUND", "error sending request", "Transport channel closed", "ECONNRESET", "ETIMEDOUT")


def _env(extra: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in ENV_ALLOW} | extra


def _outside_repos(path: Path) -> None:
    for p in [path, *path.parents]:
        if (p / ".git").exists() or (p / "CLAUDE.md").exists() or ((p / ".claude").is_dir() and p != Path.home()):
            raise RuntimeError(f"refusing to run inside {p} (repo or Claude config found)")


def _codex_home() -> Path:
    CODEX_HOME.mkdir(parents=True, exist_ok=True)
    link = CODEX_HOME / "auth.json"
    if not link.exists():
        link.symlink_to(Path.home() / ".codex/auth.json")
    for stray in ("AGENTS.md", "config.toml"):
        if (CODEX_HOME / stray).exists():
            raise RuntimeError(f"{CODEX_HOME / stray} must not exist: it would leak user instructions/config")
    return CODEX_HOME


def _claude(prompt: str, model: str, schema: dict, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="latent-gap-claude-") as wd:
        _outside_repos(Path(wd).resolve())
        cmd = ["claude", "-p", prompt, "--model", model, "--setting-sources", "", "--strict-mcp-config",
               "--mcp-config", '{"mcpServers":{}}', "--disable-slash-commands", "--tools", "",
               "--permission-mode", "dontAsk", "--no-session-persistence", "--output-format", "json",
               "--json-schema", json.dumps(schema)]
        proc = subprocess.run(cmd, cwd=wd, env=_env(CLAUDE_ISOLATION), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout)
    out = json.loads(proc.stdout) if proc.stdout.strip().startswith("{") else {}
    if proc.returncode or out.get("is_error"):
        msg = str(out.get("result") or proc.stderr or proc.stdout)[-400:]
        transient = "limit" in msg.lower() or any(t in msg for t in TRANSIENT)
        raise RateLimitedError(msg) if transient else RuntimeError(f"claude: {msg}")
    if out.get("apiKeySource", "none") not in ("none", None):
        raise RuntimeError("claude did not run on the subscription")
    if out.get("structured_output") is None:  # the model answered in prose instead (typically a refusal)
        raise RefusedError(str(out.get("result"))[:300])
    return out["structured_output"]


def _codex(prompt: str, model: str, schema: dict, timeout: float, effort: str = "medium") -> dict:
    with tempfile.TemporaryDirectory(prefix="latent-gap-codex-") as wd:
        Path(wd, "schema.json").write_text(json.dumps(schema), encoding="utf-8")
        cmd = ["codex", "exec", "--json", "--ignore-rules", "--ephemeral", "--skip-git-repo-check", "-s", "read-only",
               "-m", model, "-c", f"model_reasoning_effort={effort}", "--output-schema", "schema.json",
               "-o", "last.txt", prompt]
        proc = subprocess.run(cmd, cwd=wd, env=_env({"CODEX_HOME": str(_codex_home())}), stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode and ("status: 429" in proc.stdout + proc.stderr
                                or any(t in proc.stderr for t in TRANSIENT)):
            raise RateLimitedError(f"codex usage limit or network drop: {proc.stderr[-200:]}")
        if proc.returncode:
            raise RuntimeError(f"codex exited {proc.returncode}: {proc.stderr[-400:]}")
        return json.loads(Path(wd, "last.txt").read_text(encoding="utf-8"))


class RateLimitedError(RuntimeError):
    pass


class RefusedError(RuntimeError):
    """No structured answer: treated like a provider content-filter refusal (a failed candidate)."""


def render(messages: list[dict]) -> str:
    """Flatten a conversation into one prompt: each CLI call is stateless."""
    return "\n\n".join(f"<{m['role']}>\n{m['content']}\n</{m['role']}>" for m in messages)


class CliChat:
    """Drop-in for Gateway.chat in the decoder: `spec` is "claude:<model>" or "codex:<model>"."""

    def __init__(self, spec: str, schema: dict):
        self.cli, self.model = spec.split(":", 1)
        self.schema = schema

    def chat(self, messages: list[dict], model: str | None = None, max_tokens: int = 0, json_mode: bool = False,
             tag: str = "") -> str:
        return json.dumps(judge_call(self.cli, self.model, render(messages), self.schema))


def judge_call(cli: str, model: str, prompt: str, schema: dict, timeout: float = 600) -> dict:
    fn = _claude if cli == "claude" else _codex
    for wait in (*WAITS, None):
        try:
            return fn(prompt, model, schema, timeout)
        except RateLimitedError:
            if wait is None:
                raise
            time.sleep(wait)
