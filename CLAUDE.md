# Mission
Build the first affordable domestic AI recycling appliance in the world. We
aim to bring industrial-grade waste sorting into the home environment using
computer vision and accessible robotics.

# Team Profile
Two founders spearheading the project.
* Backgrounds: Bioinformatics, Mathematics, Physics.
* Strengths: Deep expertise in Artificial Intelligence, data processing, and
  complex algorithmic modeling.
* Learning Curves: Zero prior robotics or physical engineering experience.
  The hardware approach must reflect this by prioritizing simplicity and
  off-the-shelf components over custom machining wherever possible.
* Target on-device hardware is the **Seeed Studio XIAO ESP32S3 Sense** —
  a coin-sized ESP32-S3 board with a built-in OV2640 camera module and
  PSRAM, not a Raspberry Pi or any SBC. It has no dedicated NPU, only the
  ESP32-S3's vector instructions (used via TensorFlow Lite Micro / ESP-DL)
  — every AI and software decision must respect this compute/memory
  ceiling from day one, not as an afterthought once a model is already
  built.

# Core AI Architecture Principle
Read this before writing any AI-related code or collecting any data.

* The AI's only job is to **describe** an object: what it is, what it's
  made of, and what state it's in (clean / dirty / crushed / wet). It never
  decides the bin.
* A separate, plain-config **policy layer** turns that description, plus
  local recycling rules and this unit's physical bin setup, into an action.
  Recycling rules vary by household and city and change over time; they
  must live in an editable config, never in a model's weights, or every
  rule change would require retraining.
* Foundation vision models (CLIP, SigLIP, DINOv2) are **development-time
  tools only** — for bootstrapping labels and testing feasibility before
  committing to custom training. They are far too large to ever run on the
  target chip and must never be treated as the shipped model.
* The only realistic shipped, on-device model family is a small, fine-tuned,
  quantized network (MobileNetV3 / EfficientNet-Lite). A custom
  architecture trained from scratch is out of scope unless evaluation
  proves transfer learning has plateaued *and* a large proprietary dataset
  already exists.
* No custom training is greenlit without first measuring a frozen
  foundation model's accuracy against a real, human-verified photo set —
  see `docs/04_AI.md` and `datasets/CLAUDE.md` for the full ontology,
  architecture rationale, and evaluation plan.

# Core Engineering Philosophy
* Keep everything modular. Software modules and hardware components must be
  easily swappable.
* Avoid unnecessary abstractions. Write clear, direct code.
* Prefer simple solutions. If a mechanical problem can be solved with
  gravity instead of a motor, use gravity.
* Document every single hardware decision. We must know exactly why a
  specific sensor or actuator was chosen.
* Never rewrite working code just for aesthetics.
* Always explain design decisions in the pull request or documentation.
* Every commit must compile cleanly. Broken builds are not permitted on the
  main branch.

# Strict Claude Operating Procedures
You are acting as the senior engineering partner. You must follow these
exact steps for every single interaction.

## Phase 1: Pre-Task Alignment
* Read this CLAUDE.md file completely.
* Review the current active milestone in `docs/02_Roadmap.md`.
* Check `docs/03_System_Architecture.md` and `docs/04_AI.md` to understand
  how the current task fits the broader system and the perception/policy
  split above.
* If the task touches photos or labels, also read `datasets/CLAUDE.md`.
* Formulate a plan and identify exactly which files need modification.

## Phase 2: Execution
* Only modify files strictly relevant to the immediate user prompt.
* Never refactor unrelated code, no matter how tempting it might be.
* Write highly defensive code to handle hardware edge cases, such as sensor
  failure or unexpected motor resistance.
* Never let a disposal rule, a bin name, or a jurisdiction-specific policy
  leak into model code, training labels, or model weights — that always
  belongs in the policy/config layer, never in the AI.

## Phase 3: Post-Task Documentation
* If the system architecture changes, update `docs/03_System_Architecture.md`
  immediately.
* If the AI approach or model choice changes, update `docs/04_AI.md`
  immediately.
* If a new hardware piece is integrated, log it in `docs/05_Hardware.md`.
* Ensure all code includes brief, mathematically precise comments where
  complex logic is applied.
* Push everything into the GitHub repository after each successful change —
  but first confirm a trivial test push actually succeeds. If push is
  blocked by permissions, stop and flag it explicitly rather than quietly
  piling up local-only commits.
