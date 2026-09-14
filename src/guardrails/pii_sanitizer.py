"""PII (Personally Identifiable Information) masking and sanitization engine."""

import re
from typing import List, Tuple

class PIISanitizer:
    """Detects and redacts sensitive patterns to ensure enterprise data compliance."""

    PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "NATIONAL_ID": r"\b[12][0-9]{9}\b",  # Saudi National ID / Iqama format
        "CREDIT_CARD": r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13})\b",
        "API_TOKEN": r"\b(?:sk-[a-zA-Z0-9]{24,}|ghp_[a-zA-Z0-9]{30,}|Bearer\s+[a-zA-Z0-9_\-\.]+)\b",
        "PHONE": r"\b(?:\+?9665[0-9]{8}|05[0-9]{8}|\+?[1-9][0-9]{9,12})\b"
    }

    def __init__(self):
        self.compiled_rules = {
            category: re.compile(pattern, re.IGNORECASE)
            for category, pattern in self.PATTERNS.items()
        }

    def sanitize(self, text: str) -> Tuple[str, bool, List[str]]:
        """Redact detected PII tokens in text and return audit metadata."""
        sanitized = text
        detected_categories: List[str] = []

        for category, regex in self.compiled_rules.items():
            if regex.search(sanitized):
                sanitized = regex.sub(f"[REDACTED_{category}]", sanitized)
                detected_categories.append(category)

        modified = len(detected_categories) > 0
        return sanitized, modified, detected_categories
