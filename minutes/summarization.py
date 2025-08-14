"""Summarization functionality using Ollama API for generating meeting notes."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests


@dataclass
class OllamaConfig:
    """Configuration for Ollama summarization."""

    model: str = "llama3.1:8b"
    host: str = "http://127.0.0.1:11434"
    timeout: float = 60.0


class Summarizer:
    """Call a local Ollama reasoning model to produce minute-sized notes.

    Process flow:
    1. Receive list of transcript snippets covering ~1 minute
    2. Join transcripts and create system/user prompts
    3. Send request to local Ollama API
    4. Parse response to extract markdown bullet points
    5. Return concise meeting notes prioritizing actions/decisions
    """

    def __init__(self, cfg: OllamaConfig) -> None:
        self.cfg = cfg
        self._session = requests.Session()

    def summarize(self, texts: list[str]) -> str:
        """Summarize the list of transcript snippets into concise minute notes.

        Args:
            texts: Recent transcript snippets covering about one minute.

        Returns:
            Markdown bullet list capturing key points, decisions, and action items.
        """
        joined = "\n".join(t for t in texts if t.strip())
        if not joined.strip():
            return ""

        system = (
            "You are an expert meeting minute-taker. Generate concise, factual,\n"
            "actionable notes for the last minute of discussion. Bullets only,\n"
            "prioritize decisions, owners, due dates, blockers. Keep it under\n"
            "120 words. Use markdown list, no preamble.\n\n"
            "IMPORTANT: Try to identify different speakers and use format:\n"
            "[Person 1]: Main points from first speaker\n"
            "[Person 2]: Main points from second speaker\n"
            "If you cannot identify distinct speakers, use general bullets."
        )
        prompt = (
            "Summarize the following transcript into minute-sized notes.\n\n"
            f"Transcript:\n{joined}\n\n"
            "Return only the markdown bullets (no title)."
        )
        url = f"{self.cfg.host}/api/chat"
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        try:
            resp = self._session.post(url, json=payload, timeout=self.cfg.timeout)
            resp.raise_for_status()
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            return content
        except requests.RequestException as e:
            logging.error("Ollama request failed: %s", e)
            return ""
