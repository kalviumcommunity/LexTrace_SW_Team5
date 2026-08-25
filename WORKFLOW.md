# LexTrace Team GitHub Workflow Guidelines

This document outlines the professional GitHub workflow, branching strategy, commit conventions, code review process, and issue tracking standards for the LexTrace Data Engineering team.

---

## 🌿 1. Branching Strategy

Our team uses a **Feature Branch Workflow** to maintain pipeline stability and clear code lineage:

- **`main` Branch**: Contains production-ready, releasable code only. Direct commits to `main` are strictly prohibited.
- **Feature & Task Branches**: All development work occurs on short-lived branches created from `main`:
  - `feature/[short-description]` — New data pipelines, functions, or major enhancements (e.g., `feature/data-ingestion`)
  - `fix/[short-description]` — Bug fixes or logic corrections (e.g., `fix/validation-logic`)
  - `docs/[short-description]` — Documentation updates (e.g., `docs/data-dictionary`)
  - `refactor/[short-description]` — Code restructuring without behavior changes
  - `chore/[short-description]` — Dependency updates, tooling, or repository configuration
- **Branch Lifecycle**: Branches are merged into `main` via Pull Requests after passing review and CI checks. Once merged, feature branches are automatically deleted to keep the repository clean.

---

## 📝 2. Commit Message Conventions

All commit messages must adhere to the **Conventional Commits** standard:

### Format
```text
[type]: [short summary in imperative present tense]

[optional body explaining context and why changes were made]
```

### Commit Types Used
- `feat`: New feature or pipeline module added
- `fix`: Bug fix in logic or parsing
- `docs`: Documentation changes or data dictionary updates
- `refactor`: Code reorganization without functional side-effects
- `test`: Adding or modifying test suites
- `chore`: Tooling, dependency, or configuration updates

### Rationale
Enforcing consistent commit messages enables automated changelog generation, makes git history searchable, and allows team members to quickly understand the intent of every code modification.

---

## 🔍 3. Pull Request & Code Review Process

Pull Requests (PRs) serve as the quality gateway before code lands on `main`:

1. **Pull Request Creation**:
   - Title must clearly summarize the work (e.g., `Add data validation workflow and team branching guidelines`).
   - Description must state what changed, why, and summarize key commits.
   - Must explicitly link the parent issue using GitHub closing keywords (e.g., `Closes #2`, `Closes #3`, `Closes #4`).
2. **Review Criteria**:
   - **Correctness & Data Integrity**: Verified schema validation and data parsing logic.
   - **Clarity & Readability**: Code cleanliness, modularity, and comments.
   - **Test Coverage**: Functional verification proof included.
   - **Commit Quality**: Ensures commit history is clean and follows conventional formatting.
3. **Approval Requirements**:
   - PRs require at least **one peer approval** before merging into `main`.

---

## 📌 4. GitHub Issue Tracking Approach

Every work item in the data product pipeline starts with a GitHub Issue:

- **Issue Creation**: Every feature, bug fix, or task starts with an issue containing:
  - **Action-oriented Title**: Clear problem description (e.g., `Ingest customer transaction data into pipeline`).
  - **Contextual Description**: Explains why the work matters and details concrete acceptance criteria.
  - **Labels**: Categorizes work (e.g., `data-pipeline`, `feature`, `documentation`).
  - **Assignee**: Directly assigned to the responsible team member.
- **Traceability**: Issues are linked directly in PR descriptions so code changes remain tied to business requirement context.
- **Closure**: Issues are automatically closed when their corresponding PR is approved and merged into `main`.

---

## 📖 5. Knowledge Base Data Dictionary

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `string` | Yes | Unique document identifier |
| `title` | `string` | Yes | Document headline / subject line |
| `content` | `string` | Yes | Unstructured document body text for vector embedding |
