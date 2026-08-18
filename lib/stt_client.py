"""
lib/stt_client.py — Free Speech-to-Text via Groq Whisper

Uses Groq's free tier (whisper-large-v3-turbo) to transcribe voice messages
received from Telegram (OGG/OPUS format).

Requires: GROQ_API_KEY in environment.
Get a free key at: https://console.groq.com (no credit card required)

Usage:
    from lib.stt_client import GroqSTT

    stt = GroqSTT()
    result = stt.transcribe(ogg_bytes, filename="voice.ogg")
    if result["success"]:
        print(result["transcript"])
"""

from __future__ import annotations

import io
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3-turbo"


class GroqSTT:
    """Groq Whisper speech-to-text.

    Transcribes raw audio bytes (OGG, MP3, WAV, WEBM, FLAC, M4A)
    using the Groq Whisper API.

    Groq free tier limits (as of 2025):
      - 7,200 audio seconds / minute
      - whisper-large-v3-turbo is the recommended model
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self.api_key = (api_key or os.environ.get("GROQ_API_KEY", "")).strip()
        self.model = model or os.environ.get("STT_GROQ_MODEL", DEFAULT_MODEL)

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        filename: str = "voice.ogg",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Dict[str, object]:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (OGG/OPUS from Telegram voice messages).
            filename:    Filename hint for the API (determines MIME type detection).
                         Use "voice.ogg" for Telegram voice messages,
                         "audio.mp3" for MP3, etc.
            language:    Optional ISO-639-1 language code hint (e.g. "en", "ar").
                         Improves accuracy when language is known.
            prompt:      Optional context hint to improve transcription quality.

        Returns:
            dict with keys:
              success    bool
              transcript str   (empty string on failure)
              model      str
              provider   str   "groq"
              error      str   (only when success=False)
        """
        if not self.is_configured():
            return {
                "success": False,
                "transcript": "",
                "model": self.model,
                "provider": "groq",
                "error": (
                    "GROQ_API_KEY not configured. "
                    "Get a free key at https://console.groq.com"
                ),
            }

        if not audio_bytes:
            return {
                "success": False,
                "transcript": "",
                "model": self.model,
                "provider": "groq",
                "error": "No audio data provided.",
            }

        try:
            import requests

            files = {
                "file": (filename, io.BytesIO(audio_bytes), _mime_for(filename)),
            }
            data: Dict[str, str] = {
                "model": self.model,
                "response_format": "json",
            }
            if language:
                data["language"] = language
            if prompt:
                data["prompt"] = prompt[:224]  # Groq limit

            r = requests.post(
                GROQ_STT_URL,
                headers={"Authorization": f"Bearer {self.api_key}"},
                files=files,
                data=data,
                timeout=60,
            )

            if r.status_code == 200:
                transcript = r.json().get("text", "").strip()
                logger.info("Groq STT transcribed %d bytes → %d chars", len(audio_bytes), len(transcript))
                return {
                    "success": True,
                    "transcript": transcript,
                    "model": self.model,
                    "provider": "groq",
                }
            else:
                err = r.text[:512]
                logger.error("Groq STT HTTP %d: %s", r.status_code, err)
                return {
                    "success": False,
                    "transcript": "",
                    "model": self.model,
                    "provider": "groq",
                    "error": f"Groq API error {r.status_code}: {err}",
                }

        except Exception as e:
            logger.exception("Groq STT unexpected error")
            return {
                "success": False,
                "transcript": "",
                "model": self.model,
                "provider": "groq",
                "error": str(e),
            }


def _mime_for(filename: str) -> str:
    """Return an appropriate MIME type based on file extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    return {
        "ogg": "audio/ogg",
        "oga": "audio/ogg",
        "opus": "audio/ogg",
        "mp3": "audio/mpeg",
        "mp4": "audio/mp4",
        "m4a": "audio/mp4",
        "wav": "audio/wav",
        "webm": "audio/webm",
        "flac": "audio/flac",
    }.get(ext, "audio/ogg")
