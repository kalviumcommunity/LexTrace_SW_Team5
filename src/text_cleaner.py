import re
import unicodedata
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger("TextCleaner")

class TextCleaner:
    """
    Multi-stage text cleaning engine for RAG corpora.
    Normalizes raw extracted text from PDFs, HTML, Markdown, and TXT files into clean,
    retrieval-ready text by stripping boilerplate, fixing broken line wraps, and
    standardizing Unicode and whitespace.
    """

    # Common boilerplate regex patterns (Task 1)
    BOILERPLATE_PATTERNS = [
        # Page numbers: "Page 1 of 5", "Page 12", "- Page 3 -"
        r'(?i)^\s*[-–—]?\s*page\s+\d+(\s+of\s+\d+)?\s*[-–—]?\s*$',
        # Navigation breadcrumbs: "Home > Docs > Policy"
        r'^\s*[\w\s]+\s*>\s*[\w\s]+(\s*>\s*[\w\s]+)+\s*$',
        # Confidentiality & Copyright notices
        r'(?i)^\s*(confidential|strictly confidential|internal use only|all rights reserved|copyright\s*©.*)\s*$',
        # Document header/footer rules (e.g. repeated "LexTrace Internal Specification")
        r'(?i)^\s*lextrace\s+(internal|confidential|document)\s+(header|footer|notice)\s*$',
    ]

    def __init__(self, custom_boilerplate: Optional[List[str]] = None):
        """
        Initialize TextCleaner with baseline and optional custom boilerplate patterns.
        """
        self.compiled_boilerplate = [
            re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.BOILERPLATE_PATTERNS
        ]
        if custom_boilerplate:
            for p in custom_boilerplate:
                self.compiled_boilerplate.append(re.compile(p, re.IGNORECASE | re.MULTILINE))

    def clean_text(self, raw_text: str) -> str:
        """
        Execute full multi-stage cleaning pipeline on raw text.

        Args:
            raw_text: Uncleaned extracted text string.

        Returns:
            Normalized, clean, retrieval-ready plain text string.
        """
        if not raw_text:
            return ""

        # Stage 1: Unicode NFKC & Encoding Artifact Cleanup (Task 2)
        text = self._normalize_unicode(raw_text)

        # Stage 2: Boilerplate & Header/Footer Removal (Task 1)
        text = self._remove_boilerplate(text)

        # Stage 3: Hyphenated Line-Wrap & Broken Sentence Repair (Task 2)
        text = self._fix_line_wraps(text)

        # Stage 4: Whitespace & Line Break Normalization (Task 2)
        text = self._normalize_whitespace(text)

        return text

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode (NFKC), strip control/zero-width characters, and replace smart quotes.
        """
        # 1. Unicode Compatibility Decomposition and Recomposition (NFKC)
        text = unicodedata.normalize("NFKC", text)

        # 2. Replace non-breaking space (\xa0) and zero-width spaces (\u200b, \ufeff)
        text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")

        # 3. Standardize smart quotes and typographic dashes
        quote_replacements = {
            "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
            "‘": "'", "’": "'", "‚": "'", "`": "'",
            "–": "-", "—": "-", "―": "-"
        }
        for orig, repl in quote_replacements.items():
            text = text.replace(orig, repl)

        # 4. Remove non-printable control characters (keep \n and \t)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

        return text

    def _remove_boilerplate(self, text: str) -> str:
        """
        Remove page headers/footers, navigation text, and boilerplate disclaimers line-by-line.
        """
        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Check if line matches any boilerplate pattern
            is_boilerplate = False
            for pattern in self.compiled_boilerplate:
                if pattern.match(stripped):
                    is_boilerplate = True
                    break

            if not is_boilerplate:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _fix_line_wraps(self, text: str) -> str:
        """
        Repair hyphenated words split across line breaks (e.g. "retrie-\nval" -> "retrieval")
        and rejoin mid-sentence broken lines.
        """
        # 1a. Hyphenated word break with lower/title case fragment: "Classi-\nfication" -> "Classification", "docu-\nments" -> "documents"
        text = re.sub(r'(\b[A-Za-z]{2,})-\s*\n\s*([a-z]{2,}\b)', r'\1\2', text)
        # 1b. TitleCase compound terms across lines: "Retrieval-\nAugmented" -> "Retrieval-Augmented"
        text = re.sub(r'(\b[A-Z][a-z]{1,})-\s*\n\s*([A-Z][a-z]{1,}\b)', r'\1-\2', text)



        # 2. Join mid-sentence soft line breaks where line ends with a word/comma and next line starts with lowercase
        lines = text.splitlines()
        rejoined = []
        i = 0
        while i < len(lines):
            curr_line = lines[i]
            # If current line does not end with sentence terminal (.!?:) and next line starts lowercase
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                curr_stripped = curr_line.rstrip()
                if (
                    curr_stripped
                    and not curr_stripped.endswith((".", "!", "?", ":", ";", "#", "-"))
                    and next_line
                    and next_line[0].islower()
                    and not next_line.startswith(("- ", "* ", "1.", "2.", "3.", "4.", "5."))
                ):
                    # Join lines with a single space
                    lines[i + 1] = curr_stripped + " " + next_line
                    i += 1
                    continue
            rejoined.append(curr_line)
            i += 1

        return "\n".join(rejoined)

    def _normalize_whitespace(self, text: str) -> str:
        """
        Collapse multiple horizontal spaces, trim line trailing spaces, and collapse runaway blank lines.
        """
        # 1. Collapse multiple spaces/tabs into a single space per line
        lines = text.splitlines()
        normalized_lines = []
        for line in lines:
            clean_line = re.sub(r'[ \t]+', ' ', line).strip()
            normalized_lines.append(clean_line)

        # 2. Collapse consecutive empty lines (allow maximum of 1 empty line between paragraphs)
        result_lines = []
        empty_count = 0
        for line in normalized_lines:
            if not line:
                empty_count += 1
                if empty_count <= 1:
                    result_lines.append("")
            else:
                empty_count = 0
                result_lines.append(line)

        return "\n".join(result_lines).strip()

    def compare(self, raw_text: str, cleaned_text: str) -> Dict[str, Any]:
        """
        Generate before/after statistics and diff summary for confirmation evidence (Task 4).
        """
        raw_len = len(raw_text)
        cleaned_len = len(cleaned_text)
        raw_words = len(raw_text.split())
        cleaned_words = len(cleaned_text.split())

        char_diff = raw_len - cleaned_len
        char_reduction_pct = (char_diff / raw_len * 100) if raw_len > 0 else 0.0

        # Detect specific artifacts cleaned
        artifacts_detected = []
        if re.search(r'[\xa0\u200b\ufeff]', raw_text):
            artifacts_detected.append("Zero-width / Non-breaking space artifacts")
        if re.search(r'[“”‘’–—]', raw_text):
            artifacts_detected.append("Non-standard smart quotes & em-dashes")
        if re.search(r'(?i)page\s+\d+(\s+of\s+\d+)?', raw_text):
            artifacts_detected.append("Header / Footer page numbers")
        if re.search(r'(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)', raw_text):
            artifacts_detected.append("Broken hyphenated line wraps")
        if re.search(r'\n{3,}', raw_text):
            artifacts_detected.append("Runaway blank lines")

        return {
            "raw_char_count": raw_len,
            "cleaned_char_count": cleaned_len,
            "raw_word_count": raw_words,
            "cleaned_word_count": cleaned_words,
            "char_reduction_count": char_diff,
            "char_reduction_pct": round(char_reduction_pct, 2),
            "artifacts_detected": artifacts_detected
        }
