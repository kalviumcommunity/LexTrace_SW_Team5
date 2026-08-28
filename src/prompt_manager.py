"""
Prompt Template Engine & Manager Module for LexTrace RAG Assistant
Decouples prompt text definitions from Python application code.
Supports template loading, versioning, named placeholder validation, and runtime value injection.
"""
import string
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

# Ensure root directory resolution
sys_path_root = Path(__file__).resolve().parent.parent
from src.config import Config

logger = logging.getLogger("LexTrace.PromptManager")

class PromptTemplate:
    """
    Encapsulates a reusable prompt template with named placeholders.
    Handles placeholder discovery, validation, and safe runtime rendering.
    """
    def __init__(self, name: str, template_text: str, version: str = "1.0", file_path: Optional[Path] = None):
        self.name = name
        self.template_text = template_text.strip()
        self.version = version
        self.file_path = file_path
        self.placeholders: Set[str] = self.extract_placeholders()

    def extract_placeholders(self) -> Set[str]:
        """Inspects template text and extracts all unique named placeholders."""
        formatter = string.Formatter()
        placeholders = set()
        for _, field_name, _, _ in formatter.parse(self.template_text):
            if field_name is not None and field_name != "":
                # Extract key if nested (e.g., 'user.name' -> 'user')
                key = field_name.split('.')[0].split('[')[0]
                placeholders.add(key)
        return placeholders

    def render(self, **kwargs) -> str:
        """
        Injects dynamic values at runtime into named placeholders.
        Validates that all required placeholders are supplied before rendering.
        """
        missing = [p for p in self.placeholders if p not in kwargs]
        if missing:
            err_msg = f"Cannot render template '{self.name}' (v{self.version}): Missing required placeholder(s): {missing}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        try:
            rendered = self.template_text.format(**kwargs)
            logger.debug(f"Rendered template '{self.name}' (v{self.version}) successfully.")
            return rendered
        except Exception as e:
            err_msg = f"Failed to render template '{self.name}': {str(e)}"
            logger.error(err_msg)
            raise RuntimeError(err_msg)


class PromptManager:
    """
    Central repository loader and manager for system & user prompt templates.
    """
    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or Config.PROMPTS_DIR
        self._cache: Dict[str, PromptTemplate] = {}

    def get_template(self, template_name: str, filename: Optional[str] = None) -> PromptTemplate:
        """
        Retrieves a cached template or loads it from the prompts/ directory.
        """
        if template_name in self._cache:
            return self._cache[template_name]

        target_file = None
        if filename:
            target_file = self.prompts_dir / filename
        else:
            # Try conventional filenames
            candidates = [
                self.prompts_dir / f"{template_name}_template.txt",
                self.prompts_dir / f"{template_name}.txt",
                self.prompts_dir / f"system_prompt_{template_name}.txt"
            ]
            for c in candidates:
                if c.exists():
                    target_file = c
                    break

        if not target_file or not target_file.exists():
            raise FileNotFoundError(f"Prompt template file for '{template_name}' not found in {self.prompts_dir}")

        text = target_file.read_text(encoding="utf-8")
        template = PromptTemplate(name=template_name, template_text=text, version="1.0", file_path=target_file)
        self._cache[template_name] = template
        logger.info(f"Loaded prompt template '{template_name}' from {target_file.name} (Placeholders: {list(template.placeholders)})")
        return template

    def render_prompt(self, template_name: str, **kwargs) -> str:
        """
        Convenience method to load and render a template in a single invocation.
        """
        template = self.get_template(template_name)
        return template.render(**kwargs)


# Global default manager instance
prompt_manager = PromptManager()
