# Dataset Labeling Protocol

This file governs how raw photos in `datasets/raw/` get turned into rows in
`datasets/labels/manifest.xlsx`. Read this completely before touching
photos or labels.

## Source of truth for material vs. object/state
* If photos arrive pre-sorted into per-material folders (as the first batch
  did — cardboard/glass/metal/paper/plastic/trash) and the folder name is
  itself a valid value in the material vocabulary below, it is trusted as
  the `material` label and not re-guessed per photo. A folder named
  something outside the vocabulary (e.g. a generic "trash" catch-all) is
  *not* trusted as-is — material for those photos is assessed from the
  image like any other field.
* `object` and `state` always require looking at the actual image — they
  are never inferred from filename or folder.
* Photos are moved (not copied) into `datasets/raw/` flat, preserving
  original filenames exactly. A subfolder (e.g. `datasets/raw/session1/`)
  is fine too if that keeps a batch tidier — either layout is allowed.

## Batching
* Process photos in batches of 15-30, never all at once.
* Append each finished batch's rows to `datasets/labels/manifest.xlsx`
  immediately — do not hold everything until the end.

## Columns — exactly these four, no others, no bin column ever
1. **photo filename** — must match the file in `datasets/raw/` exactly.
2. **object** — recycling-oriented vocabulary: bottle, jar, cup, box, can,
   film, produce, unknown, ... (open vocabulary, but stay in this style;
   use `unknown` rather than guessing wildly).
3. **material** — coarse only: plastic, paper, cardboard, glass, metal,
   organic, composite, unknown. Never a fine-grained resin code (no "PET",
   no "HDPE").
4. **state** — clean, food residue, liquid, crushed, wet, mixed. Can
   combine more than one state per photo where visibly true.

## What this is NOT
* These are best-guess labels, not verified ground truth. Nothing gets
  marked verified or final by the labeling pass itself — human review is a
  separate, later step done by hand.
* No disposal rule, bin name, or jurisdiction-specific policy ever appears
  here — see the policy/config split in the root `CLAUDE.md`. This file is
  purely about describing the object, never about routing it.
* This protocol does not license inventing values outside the object/
  material/state vocabulary above.

See `docs/04_AI.md` for how this manifest feeds the perception-model
evaluation plan, and the root `CLAUDE.md` for the AI architecture
principle this all serves.
