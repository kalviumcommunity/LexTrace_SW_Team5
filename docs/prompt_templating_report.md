# Technical Report: Reusable Prompt Templating & Logic Separation Architecture for LexTrace RAG Assistant

## Executive Summary

As enterprise LLM applications expand across multiple features (e.g. interactive chat, batch processing, background compliance auditing, and API endpoints), hardcoding prompt strings directly inside Python source code creates severe engineering bottlenecks:
- **Prompt Duplication & Drift**: Minor wording changes or safety constraint updates require editing identical string literals across dozens of files.
- **Tight Coupling**: Non-engineering prompt engineers or domain experts cannot update prompt wording without modifying Python source files and triggering full application deployment cycles.
- **Inconsistent Output Quality**: Variations in inline system prompts cause inconsistent grounding, tone, and formatting across features.

This report details the architectural refactoring of the LexTrace Internal RAG Assistant to centralize prompt templates with named placeholders in `prompts/`, decoupled from Python business logic, and managed by a versioned `PromptManager` engine.

---

## 1. Architectural Overview & Logic Decoupling (Task 4)

Prompt definitions are completely isolated from Python application logic:

```
LexTrace_SW_Team5/
├── prompts/                         <-- Pure Text Template Repository (Git Tracked)
│   ├── system_prompt_template.txt   # System prompt ({role}, {domain}, {constraints})
│   ├── rag_query_template.txt       # RAG user query ({context}, {question}, {format_instructions})
│   └── batch_audit_template.txt     # Batch document audit ({document_id}, {document_text}, {audit_criteria})
├── src/                             <-- Python Application Source Code
│   ├── prompt_manager.py            # Central Template Manager & Validation Engine
│   ├── chat_completion.py           # Feature 1: Interactive RAG Chat (Consumes templates)
│   └── batch_prompt_runner.py       # Feature 2: Batch Document Audit CLI (Consumes templates)
└── outputs/                         <-- Evaluation & Render Logs
    ├── prompt_template_renders.log
    └── prompt_template_renders.json
```

### Key Architectural Benefits:
1. **Single Source of Truth**: Wording changes made in `prompts/rag_query_template.txt` instantly update prompt behavior across all features.
2. **Zero-Code Prompt Iteration**: Prompt engineers can refine guidelines or formatting instructions without touching Python code or risking logic bugs.
3. **App-Wide Consistency**: Guaranteed uniform tone, safety guidelines, and output schemas across interactive and batch features.

---

## 2. Named Placeholders & Runtime Injection Mechanics (Task 1 & 2)

Templates use explicit `{placeholder_name}` tags syntax. 

### 2.1 Example RAG Query Template (`prompts/rag_query_template.txt`)
```text
Retrieved Document Context:
---
{context}
---

Staff User Question:
{question}

Formatting Instructions:
{format_instructions}
```

### 2.2 Dynamic Runtime Value Injection
At request time, Python feature modules pass dynamic runtime values to `PromptManager.render_prompt()`:

```python
from src.prompt_manager import PromptManager

pm = PromptManager()
rendered_prompt = pm.render_prompt(
    "rag_query",
    context=retrieved_vector_docs,
    question=user_input_query,
    format_instructions="Summarize in 2 bullet points with citations."
)
```

### 2.3 Placeholder Validation & Error Handling
`PromptTemplate.extract_placeholders()` automatically parses `{name}` tags. Before rendering, `render()` verifies that all required placeholders exist in the runtime `kwargs`. If a required key is omitted, it raises a `ValueError` with clear diagnostics instead of sending broken prompts to the LLM.

---

## 3. Multi-Feature Prompt Reuse (Task 3)

The centralized `PromptManager` engine is reused across two distinct application entry points:

### Feature 1: Interactive RAG Chat (`src/chat_completion.py`)
Used for real-time staff queries. Dynamically renders `system_prompt_template.txt` and `rag_query_template.txt` with user question context.

```python
system_prompt = pm.render_prompt(
    "system_prompt",
    role="LexTrace Internal RAG Assistant",
    domain="Workplace Policy",
    constraints="Do not hallucinate policy facts."
)
```

### Feature 2: Batch Document Audit CLI (`src/batch_prompt_runner.py`)
Used for offline compliance validation of knowledge base records. Reuses `system_prompt_template.txt` and `batch_audit_template.txt` across batch document iteration.

```python
rendered_audit_prompt = pm.render_prompt(
    "batch_audit",
    document_id=doc_id,
    title=title,
    document_text=text,
    audit_criteria="Check for 30-day filing deadlines."
)
```

---

## 4. Template Versioning & Safe Update Strategy (Video Q&A)

When prompts must change as the application scales, updates must be performed safely without breaking existing features:

1. **Semantic Versioning for Templates**: Assign explicit version metadata (e.g. `v1.0`, `v2.0`) to template files.
2. **Backward Compatible Placeholders**: When adding new placeholders, supply optional default fallback values in `PromptTemplate` to prevent breaking legacy feature calls that omit the new variable.
3. **Staging & Parallel Template Testing**: Maintain side-by-side template versions (e.g. `rag_query_v1.txt` and `rag_query_v2.txt`) during A/B testing or prompt migration.
4. **Automated Validation Integration**: Run automated render tests (`src/run_template_demo.py`) in CI/CD pipelines to verify that all placeholders render cleanly before merging PRs.
