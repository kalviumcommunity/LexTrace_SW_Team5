# Technical Report: Structured JSON Output Prompting & Resilient Parsing Strategy for LexTrace RAG Assistant

## Executive Summary

In production Retrieval-Augmented Generation (RAG) applications, large language models (LLMs) do not operate in isolation. Downstream application components—such as UI rendering engines, citation linkers, database loggers, and security audit systems—require programmatic access to distinct response fields (e.g., answer text, document citations, and model confidence scores).

Free-form prose responses cannot be parsed safely by application code. Unstructured completions lead to parsing errors, broken UI links, and unhandled runtime exceptions.

This report details the implementation of structured JSON response prompting (`response_format={"type": "json_object"}`), a multi-tier resilient JSON parsing engine, and schema field validation for the LexTrace Internal RAG Assistant.

---

## 1. Why Structured Output is Required for App Integration

When an enterprise app queries a RAG assistant, it must perform downstream operations based on the returned response:
- **UI Citation Rendering**: Display clickable links to specific internal HR/Finance documents.
- **Audit & Compliance Logging**: Log model confidence scores and cited policy IDs to data warehouses.
- **Automated Workflow Execution**: Trigger automated ticket routing based on structured policy categories.

Attempting to extract citations or confidence scores from free-form text using regex or string splitting is brittle. Subtle changes in formatting (e.g. bolding, line breaks, conversational preamble) cause parsing failures. Returning structured JSON guarantees programmatic reliability.

---

## 2. JSON Response Prompting & API Configuration (Task 1)

### 2.1 API Configuration
We configure the API request with `response_format={"type": "json_object"}` in OpenAI-compatible client completions:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.0,
    response_format={"type": "json_object"}
)
```

### 2.2 System Prompt Schema Definition
To ensure valid JSON generation, the system prompt explicitly specifies the required JSON schema and output rules:

```text
CRITICAL REQUIREMENT: You MUST respond ONLY with a single valid JSON object.
The JSON object MUST strictly adhere to the following schema:
{
  "answer": "Factual policy summary string.",
  "sources": ["List of internal document names or portal URLs cited."],
  "confidence": 0.95
}
Rules: Output valid JSON ONLY without preamble or conversational text.
```

---

## 3. Resilient Multi-Tier Parsing Architecture (Task 2 & 3)

LLMs occasionally return JSON enclosed in markdown code fences (` ```json `), preceded by introductory prose, or with minor syntax flaws. 

To prevent unhandled exceptions or crashes, `src/structured_parser.py` implements a **4-Tier Resilient Parsing Engine**:

```
+-----------------------------------------------------------------------+
|                       Raw LLM Completion String                        |
+-----------------------------------------------------------------------+
                                   |
                                   v
             [Tier 1: Direct JSON Parse (json.loads)] 
                                   | (Success)
                     +-------------+-------------+
                     |                           | (Failure)
                     v                           v
          { Return Dict Object }     [Tier 2: Markdown Codeblock Regex]
                                                 | (Success)
                                   +-------------+-------------+
                                   |                           | (Failure)
                                   v                           v
                        { Return Dict Object }     [Tier 3: Substring Regex ({...})]
                                                               | (Success)
                                                 +-------------+-------------+
                                                 |                           | (Failure)
                                                 v                           v
                                      { Return Dict Object }     [Tier 4: Malformed Catch]
                                                                     (Report Error Gracefully)
```

### Parsing Tiers Detail:
1. **Tier 1 (Direct JSON Parse)**: Attempts `json.loads(cleaned_str)`. Fastest path for pure JSON completions.
2. **Tier 2 (Markdown Codeblock Extraction)**: Uses regex `r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"` to isolate JSON content inside code fences.
3. **Tier 3 (Regex Substring Extraction)**: Uses regex `r"(\{[\s\S]*\})"` to locate the outermost `{ ... }` structure in surrounding text, and sanitizes trailing commas before closing braces.
4. **Tier 4 (Graceful Malformed Catch)**: If all extraction tiers fail, returns `parsed_successfully: False` with `error_type: "UNPARSEABLE_JSON_SYNTAX"` without crashing the application.

---

## 4. Required Field Schema Validation & Auto-Remediation (Task 4)

Parsing raw JSON into a Python dictionary is necessary but insufficient. The dictionary must be validated to ensure required keys exist and conform to expected types.

`validate_and_normalize_schema()` enforces validation rules:

### 4.1 Schema Rules & Field Normalization

| Field Name | Type | Required | Validation & Auto-Remediation Rule |
|---|---|---|---|
| `answer` | `str` | **Yes** | Must be a non-empty string. If missing, the response is rejected with status `"MISSING_REQUIRED_FIELDS"`. |
| `sources` | `list[str]` | **Yes** | Must be a list of strings. **Auto-Remediation**: If model returns singular `"source"`, normalizes into `"sources": [source]`. If missing, populates default system citation with warning. |
| `confidence` | `float` | No | Float between 0.0 and 1.0. Assigned default `0.90` if omitted by model. |

### 4.2 Empirical Validation Results

| Scenario | Input Condition | Parsing Result | Validation Status | App Behavior |
|---|---|---|---|---|
| **Scenario 1** | Pure JSON Payload | Tier 1 Parse | `VALIDATED_CLEAN` | Direct downstream consumption. |
| **Scenario 2** | Markdown Wrapped (` ```json `) | Tier 2 Extracted | `VALIDATED_CLEAN` | Recovered and consumed cleanly. |
| **Scenario 3** | Broken Unquoted Syntax | Tier 4 Handled | `UNPARSEABLE_JSON_SYNTAX` | Caught gracefully; logged error without crashing. |
| **Scenario 4** | Singular `"source"` Alias | Tier 1 Parse | `VALIDATED_WITH_WARNINGS` | Auto-remediated to `"sources": [...]`. |
| **Scenario 5** | Missing `"answer"` Key | Tier 1 Parse | `MISSING_REQUIRED_FIELDS` | Rejected cleanly with actionable error diagnosis. |

---

## 5. Recovery Strategy for Malformed Output (Video Q&A)

When building production RAG systems, malformed model output must be handled through a layered recovery pipeline:

1. **Layer 1: Resilient Multi-Tier Local Parsing**: Attempt regex extraction and syntax sanitization (as implemented in `StructuredResponseParser`).
2. **Layer 2: Fallback Field Remediation**: Map alias keys (`source` -> `sources`) or populate default fallback citations.
3. **Layer 3: Automated LLM Re-prompting (Repair Prompt)**: If parsing or validation fails critically, pass the malformed text and JSON schema error back to the LLM with a target repair prompt:
   *"Your previous response failed JSON validation with error: <error>. Re-format the following content into valid JSON strictly adhering to schema..."*
4. **Layer 4: Deterministic Fallback Object**: If re-prompting fails or times out, return a safe fallback dictionary object (`{"answer": "<raw text>", "sources": ["Unverified"], "status": "fallback"}`) to preserve user experience without crashing.
