"""
Data Validation Module for LexTrace RAG Assistant
Validates incoming CSV/JSON documents against required schema specifications.
"""
import json
from pathlib import Path
from typing import Dict, List, Any

REQUIRED_FIELDS = {"id", "title", "content"}

def validate_dataset(file_path: Path) -> Dict[str, Any]:
    """
    Validates a dataset file for schema completeness and encoding.
    Returns a dictionary summarizing validation results.
    """
    if not file_path.exists():
        return {"valid": False, "error": f"File not found: {file_path}"}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return {"valid": False, "error": "Dataset must be a list of records"}

        invalid_records = 0
        for i, record in enumerate(data):
            if not isinstance(record, dict) or not REQUIRED_FIELDS.issubset(record.keys()):
                invalid_records += 1

        is_valid = invalid_records == 0
        return {
            "valid": is_valid,
            "total_records": len(data),
            "invalid_records": invalid_records,
            "error": None if is_valid else f"{invalid_records} records failed schema validation"
        }
    except Exception as e:
        return {"valid": False, "error": f"Encoding/Parsing error: {str(e)}"}
