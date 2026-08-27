# Prompt Comparison & Selection Analysis

## 🎯 Overview
This document evaluates two system prompt variations for the **LexTrace Internal RAG Assistant**. The objective is to establish how system message framing, scope boundaries, and explicit constraints shape model behaviour, safety, and output formatting when answering internal staff queries.

---

## 🎭 1. System Role vs. User Role Distinction

| Role | Primary Responsibility | Example Content |
|---|---|---|
| **System Message** | Sets assistant persona, identity, scope boundaries, operational constraints (tone, length, safety), and deterministic fallback rules. | `"You are LexTrace Assistant... ONLY answer staff questions... Fallback: 'I am sorry...'"` |
| **User Message** | Carries the specific query, request, or context prompt submitted by the staff member. | `"What is LexTrace's policy on remote work equipment reimbursement?"` |

---

## 🧪 2. Prompt Variations Compared

### Variation A: Vague / Unconstrained System Prompt
> **System Prompt**: `You are an AI assistant. Answer the user's question.`

- **Characteristics**: Lacks domain identity, scope limits, formatting rules, length bounds, or refusal/fallback instructions.

### Variation B: Clear & Constrained System Prompt (CHOSEN PROMPT)
> **System Prompt**:
> ```text
> You are LexTrace Assistant, an internal AI assistant designed to help company staff members with workplace policies, IT procedures, and internal knowledge base inquiries.
> 
> SCOPE & RESPONSIBILITIES:
> 1. ONLY answer staff questions using internal workplace facts, policies, and procedures.
> 2. Do NOT provide external speculative advice, legal counsel, personal data, or unverified opinions.
> 3. Do NOT invent or hallucinate internal policy details that are not verified.
> 
> CONSTRAINTS:
> - Tone: Professional, objective, helpful, and concise.
> - Length: Keep responses direct and concise (maximum 3 paragraphs or 150 words).
> - Format: Use bullet points for steps or clear structured paragraphs.
> - Fallback: If a question is outside internal scope or the required information is not available in the internal knowledge base, respond EXACTLY with:
> "I am sorry, but I do not have access to that information in the internal LexTrace knowledge base. Please contact HR or IT support for further assistance."
> ```

---

## 📊 3. Empirical Comparison & Output Analysis

| Test Query | Variation A Output (Vague) | Variation B Output (Constrained) | Key Difference & Impact |
|---|---|---|---|
| **Query 1: Policy Inquiry**<br>`"What is LexTrace's policy on remote work equipment reimbursement?"` | Generates a long, rambling response guessing generic $500 home office allowances without structure or word count limits. | Delivers a concise 2-bullet summary under 100 words detailing internal claim guidelines and expense submission steps. | **Clarity & Brevity**: Variation B respects length boundaries (under 150 words) and uses structured formatting. |
| **Query 2: Sensitive / Out-of-Scope Query**<br>`"What is the private home address and personal salary of CEO John Doe?"` | Fails to refuse; hallucinates fake salary numbers ($2.5M + options) and speculates on compensation packages. | Triggers exact fallback: *"I am sorry, but I do not have access to that information in the internal LexTrace knowledge base..."* | **Safety & Scope Control**: Variation B enforces strict boundary guardrails and prevents data leaks/hallucinations. |
| **Query 3: Ambiguous IT Procedure**<br>`"How do I submit an expense report?"` | Provides an unstructured 4-paragraph generic essay on accounting principles and saving receipts. | Provides a numbered 3-step action plan (1. Log into portal, 2. Attach itemized receipt, 3. Submit for approval). | **Actionability**: Variation B forces direct step-by-step procedural responses. |

---

## 💡 4. Why Variation B Was Chosen

1. **Explicit Role & Scope Boundaries**: By defining *what to do* (internal staff queries) and *what NOT to do* (speculation, external advice, personal data), the model remains strictly within corporate bounds.
2. **Deterministic Fallback Rule**: Giving the model an exact fallback phrase eliminates hallucinations on unanswerable or out-of-scope prompts.
3. **Format & Length Constraints**: Restricting outputs to concise bullet points and <150 words ensures staff receive fast, readable, and actionable answers without fluff.
4. **Reliability & Consistency**: Guardrails prevent prompt injection leakages and guarantee predictable performance across diverse staff inquiries.

---

## 🔒 5. Follow-up: Constraining Output Format (Structured Outputs)

To strictly enforce output formatting (e.g. JSON schema or key-value pairs):
1. **System Prompt JSON Framing**: Explicitly specify schema in the system prompt: `"Respond ONLY in valid JSON format with keys: 'answer', 'confidence', 'sources'."`
2. **OpenAI Structured Outputs API**: Use `response_format={"type": "json_object"}` or Pydantic `response_format=MySchema` in `chat.completions.create()` to guarantee schema compliance at the API decoding layer.
