# llm_wrapper.py

import subprocess, shlex, json, time, os
from typing import Tuple

USE_OLLAMA = True  # flip True if you have Ollama CLI and model installed
OLLAMA_MODEL = "mistral"  # change to your local model name

def call_ollama(prompt: str, temperature: float = 0.0, max_tokens: int = 256, timeout: int = 10) -> str:
    cmd = f"ollama run {shlex.quote(OLLAMA_MODEL)} --temperature {temperature} --n 1 --max-tokens {max_tokens} --prompt {shlex.quote(prompt)}"
    try:
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            return proc.stdout.strip()
        else:
            return f"[OLLAMA_ERR] {proc.stderr.strip()}"
    except Exception as e:
        return f"[OLLAMA_EXCEPTION] {e}"

def rule_based_fallback(prompt: str) -> str:
    # VERY simple deterministic decision generator for offline testing.
    # Looks for keywords and returns a JSON-like string.
    out = {"SELL_PERCENT": 1.0, "INVEST_TRACE": "NO", "Rationale": "fallback-rule: sell if inventory present"}
    if "artisanal" in prompt.lower():
        out["SELL_PERCENT"] = 1.0
        out["INVEST_TRACE"] = "NO"
    if "industrial" in prompt.lower():
        out["SELL_PERCENT"] = 0.5
        out["INVEST_TRACE"] = "YES"
    return json.dumps(out)

def llm_decide(prompt: str, temperature: float = 0.0, max_tokens: int = 256) -> Tuple[str, float]:
    """
    Returns (llm_response_text, elapsed_seconds)
    """
    t0 = time.time()
    if USE_OLLAMA:
        resp = call_ollama(prompt, temperature=temperature, max_tokens=max_tokens)
    else:
        resp = rule_based_fallback(prompt)
    return resp, time.time() - t0
    