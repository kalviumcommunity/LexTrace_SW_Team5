# LexTrace Team GitHub Workflow Guidelines

This document outlines the professional GitHub workflow, branching strategy, commit conventions, code review process, and issue tracking standards for the LexTrace Data Engineering team.

---

## 🌿 Branching Strategy

Our team uses a **Feature Branch Workflow** to maintain pipeline stability and clear code lineage:

- **`main` Branch**: Contains production-ready, releasable code only. Direct commits to `main` are strictly prohibited.
- **Feature & Task Branches**: All development work occurs on short-lived branches created from `main`:
  - `feature/[short-description]` — New data pipelines, functions, or major enhancements (e.g., `feature/data-ingestion`)
  - `fix/[short-description]` — Bug fixes or logic corrections (e.g., `fix/validation-logic`)
  - `docs/[short-description]` — Documentation updates (e.g., `docs/data-dictionary`)
  - `refactor/[short-description]` — Code restructuring without behavior changes
  - `chore/[short-description]` — Dependency updates, tooling, or repository configuration
- **Branch Lifecycle**: Branches are merged into `main` via Pull Requests after passing review and CI checks. Once merged, feature branches are automatically deleted to keep the repository clean.
