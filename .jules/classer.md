# Classer Journal — ComfyUI-fal-API

*Transforming procedural FAL integration into elegant Python classes.*

---

## 2026-02-25 - FileUploadCache Extraction

**Learning:** Static utility classes (`ImageUtils`) often accumulate cross-cutting concerns. Cache management with JSON persistence is a prime extraction candidate — it deserves its own cohesive class, not scattered class variables in a utility module.

**Action:** Extract `FileUploadCache` as a dedicated class. This improves testability, makes the cache reusable elsewhere, and clarifies ImageUtils' actual responsibilities (tensor conversion, image preprocessing). Low-risk refactor with immediate clarity gain.

---
