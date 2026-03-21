"""
SkillMind Sanitizer — anonymize and redact sensitive data before storing memories.

Prevents leaking:
- API keys, tokens, passwords
- Email addresses, phone numbers
- IP addresses, URLs with credentials
- Personal names (optional, configurable)
- Financial data (IBAN, credit card numbers)
- Custom patterns (regex-configurable)

Usage:
    sanitizer = Sanitizer()
    clean = sanitizer.sanitize("My API key is sk-ant-abc123...")
    # -> "My API key is [REDACTED:API_KEY]"

    sanitizer = Sanitizer(redact_names=["Antonio Blago", "Joerg Zimmer"])
    clean = sanitizer.sanitize("Antonio Blago called Joerg Zimmer")
    # -> "[PERSON_1] called [PERSON_2]"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SanitizeResult:
    """Result of sanitization."""
    original: str
    sanitized: str
    redactions: list[dict[str, str]]  # [{"type": "API_KEY", "original": "sk-ant-...", "replacement": "[REDACTED:API_KEY]"}]
    was_modified: bool

    @property
    def redaction_count(self) -> int:
        return len(self.redactions)


# ── Pattern definitions ───────────────────────────────────────

# API keys and tokens
API_KEY_PATTERNS: list[tuple[str, str]] = [
    # Anthropic
    (r'sk-ant-[a-zA-Z0-9_-]{20,}', "ANTHROPIC_KEY"),
    # OpenAI
    (r'sk-[a-zA-Z0-9]{20,}', "OPENAI_KEY"),
    # Pinecone
    (r'pcsk_[a-zA-Z0-9_-]{20,}', "PINECONE_KEY"),
    # Supabase
    (r'eyJ[a-zA-Z0-9_-]{50,}\.eyJ[a-zA-Z0-9_-]+', "JWT_TOKEN"),
    # Generic bearer tokens
    (r'Bearer\s+[a-zA-Z0-9_.-]{20,}', "BEARER_TOKEN"),
    # Generic API keys (key=value patterns)
    (r'(?:api[_-]?key|apikey|token|secret|password)\s*[=:]\s*["\']?([a-zA-Z0-9_.-]{16,})["\']?', "API_KEY"),
    # GitHub tokens
    (r'gh[ps]_[a-zA-Z0-9]{36,}', "GITHUB_TOKEN"),
    (r'github_pat_[a-zA-Z0-9_]{22,}', "GITHUB_PAT"),
    # AWS
    (r'AKIA[0-9A-Z]{16}', "AWS_ACCESS_KEY"),
    # Slack
    (r'xox[baprs]-[a-zA-Z0-9-]{10,}', "SLACK_TOKEN"),
    # Generic long hex strings (likely secrets)
    (r'(?<![a-zA-Z0-9])[0-9a-f]{32,}(?![a-zA-Z0-9])', "HEX_SECRET"),
]

# Personal data
PERSONAL_PATTERNS: list[tuple[str, str]] = [
    # Email addresses
    (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', "EMAIL"),
    # Phone numbers (international + German)
    (r'(?:\+49|0049|\+1|001|0)\s*[\d\s/()-]{7,15}', "PHONE"),
    # German IBAN
    (r'[A-Z]{2}\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{0,2}', "IBAN"),
    # Credit card numbers
    (r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', "CREDIT_CARD"),
    # IP addresses
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', "IP_ADDRESS"),
    # URLs with credentials (user:pass@host)
    (r'https?://[^:\s]+:[^@\s]+@[^\s]+', "CREDENTIAL_URL"),
]

# File system paths with sensitive info
PATH_PATTERNS: list[tuple[str, str]] = [
    # .env file contents
    (r'[A-Z_]{3,}=\s*["\']?[a-zA-Z0-9_./+=-]{16,}["\']?', "ENV_VAR"),
    # Private key files
    (r'-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(?:RSA\s+)?PRIVATE\s+KEY-----', "PRIVATE_KEY"),
]


class Sanitizer:
    """
    Sanitize content by redacting sensitive data.

    Configurable:
    - redact_api_keys: Remove API keys, tokens (default: True)
    - redact_personal: Remove emails, phones, IBANs (default: True)
    - redact_paths: Remove env vars, private keys (default: True)
    - redact_names: List of specific names to anonymize
    - custom_patterns: Additional regex patterns to redact
    - allowlist: Patterns to NEVER redact (e.g., your public email)
    """

    def __init__(
        self,
        redact_api_keys: bool = True,
        redact_personal: bool = True,
        redact_paths: bool = True,
        redact_names: list[str] | None = None,
        custom_patterns: list[tuple[str, str]] | None = None,
        allowlist: list[str] | None = None,
    ):
        self.redact_api_keys = redact_api_keys
        self.redact_personal = redact_personal
        self.redact_paths = redact_paths
        self.redact_names = redact_names or []
        self.custom_patterns = custom_patterns or []
        self.allowlist = allowlist or []

        # Build name mapping for consistent anonymization
        self._name_map: dict[str, str] = {}
        for i, name in enumerate(self.redact_names):
            self._name_map[name.lower()] = f"[PERSON_{i+1}]"

    def sanitize(self, text: str) -> SanitizeResult:
        """
        Sanitize text by redacting all sensitive patterns.

        Returns a SanitizeResult with the cleaned text and redaction log.
        """
        redactions: list[dict[str, str]] = []
        result = text

        # 1. API keys and tokens
        if self.redact_api_keys:
            for pattern, label in API_KEY_PATTERNS:
                result, new_redactions = self._apply_pattern(result, pattern, label)
                redactions.extend(new_redactions)

        # 2. Personal data
        if self.redact_personal:
            for pattern, label in PERSONAL_PATTERNS:
                result, new_redactions = self._apply_pattern(result, pattern, label)
                redactions.extend(new_redactions)

        # 3. Path/env patterns
        if self.redact_paths:
            for pattern, label in PATH_PATTERNS:
                result, new_redactions = self._apply_pattern(result, pattern, label)
                redactions.extend(new_redactions)

        # 4. Named persons
        for name, replacement in self._name_map.items():
            # Case-insensitive name replacement
            pattern = re.compile(re.escape(name), re.IGNORECASE)
            matches = pattern.findall(result)
            if matches:
                for match in set(matches):
                    redactions.append({
                        "type": "PERSON_NAME",
                        "original": match,
                        "replacement": replacement,
                    })
                result = pattern.sub(replacement, result)

        # 5. Custom patterns
        for pattern, label in self.custom_patterns:
            result, new_redactions = self._apply_pattern(result, pattern, label)
            redactions.extend(new_redactions)

        return SanitizeResult(
            original=text,
            sanitized=result,
            redactions=redactions,
            was_modified=result != text,
        )

    def sanitize_memory_content(self, content: str) -> str:
        """Quick sanitize — returns just the cleaned string."""
        return self.sanitize(content).sanitized

    def _apply_pattern(
        self, text: str, pattern: str, label: str
    ) -> tuple[str, list[dict[str, str]]]:
        """Apply a single regex pattern and collect redactions."""
        redactions: list[dict[str, str]] = []
        replacement = f"[REDACTED:{label}]"

        try:
            matches = list(re.finditer(pattern, text))
        except re.error:
            return text, redactions

        for match in reversed(matches):  # Reverse to preserve positions
            original = match.group(0)

            # Check allowlist
            if any(allowed in original for allowed in self.allowlist):
                continue

            redactions.append({
                "type": label,
                "original": original[:20] + "..." if len(original) > 20 else original,
                "replacement": replacement,
            })
            text = text[:match.start()] + replacement + text[match.end():]

        return text, redactions

    def get_stats(self, result: SanitizeResult) -> dict[str, Any]:
        """Get summary statistics of redactions."""
        by_type: dict[str, int] = {}
        for r in result.redactions:
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        return {
            "total_redactions": result.redaction_count,
            "was_modified": result.was_modified,
            "by_type": by_type,
        }


# ── Convenience function ──────────────────────────────────────

def create_default_sanitizer() -> Sanitizer:
    """Create a sanitizer with sensible defaults."""
    return Sanitizer(
        redact_api_keys=True,
        redact_personal=True,
        redact_paths=True,
    )
