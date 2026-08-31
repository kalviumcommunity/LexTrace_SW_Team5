import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DocumentLoader")

# Third-party imports with fallback warnings
try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    logger.warning("pypdf package not installed. PDF parsing will fail gracefully.")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 package not installed. HTML tag stripping will use fallback regex.")


from src.text_cleaner import TextCleaner

@dataclass
class LoadedDocument:
    """
    Standardized plain-text representation of an ingested document for RAG indexing.
    Preserves source identity metadata and loading status.
    """
    source_id: str                          # Relative path / filename for downstream citation
    source_path: str                        # Absolute filesystem path
    file_type: str                          # Standardized format tag: pdf, html, md, txt
    text_content: str = ""                  # Clean, normalized plain-text representation
    raw_content: str = ""                   # Raw extracted content before cleaning
    char_count: int = 0                     # Total cleaned character count
    word_count: int = 0                     # Total cleaned word count
    raw_char_count: int = 0                 # Raw uncleaned character count
    is_cleaned: bool = False                # Indicates if cleaning pipeline was applied
    status: str = "SUCCESS"                 # Loading status: SUCCESS or FAILED
    error_message: Optional[str] = None     # Descriptive error reason if status == FAILED
    loaded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        """Convert document metadata to a serializable dictionary."""
        return {
            "source_id": self.source_id,
            "source_path": self.source_path,
            "file_type": self.file_type,
            "char_count": self.char_count,
            "raw_char_count": self.raw_char_count,
            "word_count": self.word_count,
            "is_cleaned": self.is_cleaned,
            "status": self.status,
            "error_message": self.error_message,
            "loaded_at": self.loaded_at,
            "sample_snippet": self.text_content[:150] + ("..." if len(self.text_content) > 150 else "")
        }


class DocumentLoader:
    """
    Multi-format document loader for RAG pipelines.
    Converts PDFs, HTML, Markdown, and TXT files into a unified plain-text format,
    tags documents with source identifiers, applies text cleaning, and isolates failures gracefully.
    """
    
    SUPPORTED_EXTENSIONS = {
        ".txt": "txt",
        ".md": "md",
        ".markdown": "md",
        ".html": "html",
        ".htm": "html",
        ".pdf": "pdf"
    }

    def __init__(self, base_dir: Optional[Union[str, Path]] = None, clean_text: bool = True):
        """
        Initialize DocumentLoader.
        
        Args:
            base_dir: Optional base directory to compute relative source_id paths.
            clean_text: If True, applies TextCleaner pipeline to all ingested text.
        """
        self.base_dir = Path(base_dir).resolve() if base_dir else None
        self.clean_text = clean_text
        self.cleaner = TextCleaner() if clean_text else None

    def _get_source_id(self, file_path: Path) -> str:
        """Derive consistent relative source identifier or filename for citation."""
        if self.base_dir:
            try:
                return str(file_path.relative_to(self.base_dir)).replace("\\", "/")
            except ValueError:
                pass
        return file_path.name

    def load_file(self, file_path: Union[str, Path]) -> LoadedDocument:
        """
        Load a single document into plain text representation with error resilience.

        Args:
            file_path: Path to the target file.

        Returns:
            LoadedDocument dataclass containing extracted text, metadata, and status.
        """
        path = Path(file_path).resolve()
        source_id = self._get_source_id(path)
        ext = path.suffix.lower()

        # 1. Handle missing file
        if not path.exists():
            msg = f"File not found: {path}"
            logger.error(f"[{source_id}] {msg}")
            return LoadedDocument(
                source_id=source_id,
                source_path=str(path),
                file_type=ext.lstrip(".") or "unknown",
                status="FAILED",
                error_message=msg
            )

        # 2. Handle unsupported file format
        if ext not in self.SUPPORTED_EXTENSIONS:
            msg = f"Unsupported file format '{ext}'. Allowed: {list(self.SUPPORTED_EXTENSIONS.keys())}"
            logger.warning(f"[{source_id}] {msg}")
            return LoadedDocument(
                source_id=source_id,
                source_path=str(path),
                file_type=ext.lstrip(".") or "unsupported",
                status="FAILED",
                error_message=msg
            )

        file_type = self.SUPPORTED_EXTENSIONS[ext]

        # 3. Format-specific parsing with error boundaries
        try:
            if file_type == "txt":
                raw_text = self._parse_txt(path)
            elif file_type == "md":
                raw_text = self._parse_md(path)
            elif file_type == "html":
                raw_text = self._parse_html(path)
            elif file_type == "pdf":
                raw_text = self._parse_pdf(path)
            else:
                raise ValueError(f"No parser handler registered for format: {file_type}")

            raw_char_len = len(raw_text)

            # Apply TextCleaner if enabled (Task 3: Consistent Application across Corpus)
            if self.clean_text and self.cleaner:
                cleaned_text = self.cleaner.clean_text(raw_text)
                is_cleaned = True
            else:
                cleaned_text = self._normalize_text(raw_text)
                is_cleaned = False

            char_len = len(cleaned_text)
            word_count = len(cleaned_text.split())

            logger.info(f"[{source_id}] Ingested successfully ({char_len} chars, {word_count} words, cleaned={is_cleaned})")
            return LoadedDocument(
                source_id=source_id,
                source_path=str(path),
                file_type=file_type,
                raw_content=raw_text,
                text_content=cleaned_text,
                char_count=char_len,
                raw_char_count=raw_char_len,
                word_count=word_count,
                is_cleaned=is_cleaned,
                status="SUCCESS"
            )


        except Exception as e:
            error_desc = f"Parsing error ({type(e).__name__}): {str(e)}"
            logger.error(f"[{source_id}] Failed to load: {error_desc}")
            return LoadedDocument(
                source_id=source_id,
                source_path=str(path),
                file_type=file_type,
                status="FAILED",
                error_message=error_desc
            )

    def _parse_txt(self, path: Path) -> str:
        """Parse plain text document with encoding fallback."""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decoding failed for {path.name}, falling back to latin-1")
            return path.read_text(encoding="latin-1", errors="replace")

    def _parse_md(self, path: Path) -> str:
        """Parse Markdown document into plain text."""
        raw_text = self._parse_txt(path)
        # Strip markdown links/image syntax while preserving plain text readability
        clean_text = re.sub(r'!?\[([^\]]+)\]\([^\)]+\)', r'\1', raw_text)
        return clean_text

    def _parse_html(self, path: Path) -> str:
        """Parse HTML document and extract clean body text."""
        raw_html = self._parse_txt(path)
        if BS4_AVAILABLE:
            soup = BeautifulSoup(raw_html, "html.parser")
            # Remove scripts, styles, metadata
            for element in soup(["script", "style", "head", "title", "meta", "[document]"]):
                element.decompose()
            return soup.get_text(separator="\n", strip=True)
        else:
            # Simple regex fallback if bs4 is missing
            clean = re.sub(r'<script.*?>.*?</script>', '', raw_html, flags=re.DOTALL)
            clean = re.sub(r'<style.*?>.*?</style>', '', clean, flags=re.DOTALL)
            clean = re.sub(r'<[^>]+>', ' ', clean)
            return clean

    def _parse_pdf(self, path: Path) -> str:
        """Parse PDF document using PyPDF, page by page."""
        if not PYPDF_AVAILABLE:
            raise RuntimeError("pypdf dependency is not installed")
        
        reader = PdfReader(str(path))
        if len(reader.pages) == 0:
            raise ValueError("PDF file contains no pages")
            
        page_texts = []
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                page_texts.append(extracted)

        combined_text = "\n\n".join(page_texts).strip()
        if not combined_text:
            logger.warning(f"No text extracted from PDF {path.name} (might be image-only/scanned)")
        return combined_text

    def _normalize_text(self, text: str) -> str:
        """Normalize line breaks and trailing spaces for clean downstream indexing."""
        lines = [line.strip() for line in text.splitlines()]
        # Remove consecutive empty lines
        normalized = []
        empty_count = 0
        for line in lines:
            if not line:
                empty_count += 1
                if empty_count <= 1:
                    normalized.append("")
            else:
                empty_count = 0
                normalized.append(line)
        return "\n".join(normalized).strip()

    def load_directory(
        self,
        dir_path: Union[str, Path],
        recursive: bool = False
    ) -> List[LoadedDocument]:
        """
        Load all documents in a folder into plain-text LoadedDocument representations.

        Args:
            dir_path: Folder containing documents.
            recursive: If True, scan subdirectories recursively.

        Returns:
            List of LoadedDocument objects for both successful and failed files.
        """
        folder = Path(dir_path).resolve()
        if not folder.exists() or not folder.is_dir():
            logger.error(f"Directory not found: {folder}")
            return []

        if not self.base_dir:
            self.base_dir = folder

        pattern = "**/*" if recursive else "*"
        files = [p for p in folder.glob(pattern) if p.is_file()]

        logger.info(f"Discovered {len(files)} files in '{folder.name}'")
        loaded_docs: List[LoadedDocument] = []

        for f in files:
            doc = self.load_file(f)
            loaded_docs.append(doc)

        return loaded_docs

    def confirm_intake(
        self,
        docs: List[LoadedDocument],
        sample_chars: int = 150
    ) -> Dict[str, Any]:
        """
        Confirm document intake by outputting character lengths and text samples.

        Args:
            docs: List of loaded documents.
            sample_chars: Maximum characters to display for sample text.

        Returns:
            Dict containing batch statistics.
        """
        successful = [d for d in docs if d.status == "SUCCESS"]
        failed = [d for d in docs if d.status == "FAILED"]

        print("=" * 75)
        print("          DOCUMENT INTAKE VERIFICATION & INGESTION REPORT          ")
        print("=" * 75)
        print(f"Total Files Processed: {len(docs)}")
        print(f"  - Successfully Loaded : {len(successful)}")
        print(f"  - Failed / Skipped   : {len(failed)}")
        print("-" * 75)

        for i, doc in enumerate(docs, 1):
            status_tag = "[SUCCESS]" if doc.status == "SUCCESS" else "[FAILED ]"
            print(f"\n[{i:02d}] {status_tag} Source ID: {doc.source_id}")
            print(f"     Format     : {doc.file_type.upper()}")
            print(f"     File Path  : {doc.source_path}")
            
            if doc.status == "SUCCESS":
                print(f"     Text Length: {doc.char_count:,} chars | {doc.word_count:,} words")
                sample = doc.text_content[:sample_chars].replace("\n", " ")
                if len(doc.text_content) > sample_chars:
                    sample += "..."
                print(f"     Text Sample: \"{sample}\"")
            else:
                print(f"     Error Log  : {doc.error_message}")

        print("=" * 75)
        
        summary = {
            "total_processed": len(docs),
            "successful_count": len(successful),
            "failed_count": len(failed),
            "total_characters": sum(d.char_count for d in successful),
            "total_words": sum(d.word_count for d in successful),
        }
        return summary
