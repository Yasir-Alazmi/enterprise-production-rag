"""Adversarial prompt injection and system exploit detection."""

import re
from typing import Optional, Tuple

class InjectionDetector:
    """Screens incoming user inputs for jailbreaks and prompt override attempts."""

    DISALLOWED_PATTERNS = [
        (r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", "INSTRUCTION_OVERRIDE"),
        (r"disregard\s+(all\s+)?(previous|prior)\s+rules?", "RULE_DISREGARD"),
        (r"you\s+are\s+now\s+(in\s+)?(dan\s+mode|unrestricted|jailbroken)", "JAILBREAK_ATTEMPT"),
        (r"reveal\s+(your\s+)?(system\s+prompt|hidden\s+instructions)", "PROMPT_LEAK_ATTEMPT"),
        (r"</?system>", "SYSTEM_DELIMITER_INJECTION"),
        (r"\[INST\]|\[/INST\]", "LLAMA_DELIMITER_EXPLOIT"),
        (r"bypass\s+(all\s+)?safety\s+filters?", "SAFETY_BYPASS_ATTEMPT")
    ]

    def __init__(self):
        self.rules = [
            (re.compile(pat, re.IGNORECASE), code)
            for pat, code in self.DISALLOWED_PATTERNS
        ]

    def validate_prompt(self, text: str) -> Tuple[bool, Optional[str]]:
        """Validate input prompt against adversarial patterns."""
        normalized = text.strip()
        for regex, code in self.rules:
            if regex.search(normalized):
                return False, f"Potential injection attack detected: {code}"
        return True, None
