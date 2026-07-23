# Manifest labeling — session handoff

Status as of this update (2026-07-23). Background labeling agents hit an
account session-limit (resets 9:20pm Europe/Madrid) partway through batch
42–50; after that, labeling continued via direct (non-subagent) image
viewing in the main conversation, which was unaffected by the limit.

## What's done
- `datasets/raw/` holds all 2,527 photos moved from the old `training set photos/`
  folder (flat, original filenames preserved). Source folder removed.
- `datasets/CLAUDE.md` created, documenting the labeling protocol (columns,
  vocabulary, batching, the "trust folder-derived material except for the
  generic `trash` bucket" rule).
- `datasets/labels/manifest.xlsx` has **1,380 of 2,527 rows** filled in
  (batches 0–45 of an 85-batch plan, batch size 30, natural-sort order over
  the combined file list — see below for how batches were sliced).
- Fully labeled: all of `cardboard*` (403), all of `glass*` (501), all of
  `metal*` (410), `paper1.jpg`–`paper66.jpg`.
- The next unsaved file is `paper67.jpg`.

## What's NOT done yet
Remaining work: **1,147 photos** (`paper67.jpg` through the end of
`trash137.jpg`) across batches 46–84.

Note: batches 46, 47, 48, 49, 50 were originally dispatched as background
agents and **all failed** with the session-limit error before producing any
output — nothing from those five was ever saved, so this range needed (and
partly still needs) to be redone. As of this note, `paper67`–`paper216`
(batches 46-50's original scope) have NOT yet been re-labeled.

## How to resume next session
1. Confirm current row count in `datasets/labels/manifest.xlsx` (should be
   1,260 unless something changed) and confirm the last row is
   `metal326.jpg`.
2. Regenerate the remaining file-batch list: take `datasets/raw/`, sort
   filenames with a natural sort (split on digit runs, e.g.
   `cardboard1.jpg` < `cardboard2.jpg` < ... < `cardboard10.jpg`, and the
   whole list sorts into alphabetical material blocks: cardboard, glass,
   metal, paper, plastic, trash), then slice into chunks of 30. Skip the
   first 42 chunks (already done). This reproduces the exact same batch 0–84
   split used this session.
3. For each batch: view every photo, fill in `object` / `state` by looking
   at the image, and derive `material` from the filename prefix — except
   for `trash*.jpg` files, where material must be independently assessed
   (the `trash` folder is a public-dataset catch-all, not a material; expect
   composite/organic/unknown to dominate there).
4. Append each finished batch to `manifest.xlsx` immediately (don't hold
   batches in memory) — the workbook has a `photo filename, object,
   material, state` header row already in place. Note: this file was
   observed being opened in Excel mid-session, which blocks writes with a
   `PermissionError` and creates a `~$manifest.xlsx` lock file next to it —
   check for that lock file before writing, and close Excel if it's open.
5. Once every file in `datasets/raw/` has a row, do a final count check
   (2,527 rows expected) and report the material breakdown, plus every row
   whose `state` contains `[UNSURE]` — those need Roger's manual review per
   the protocol (labels here are best-guess, not verified).
6. Push to git and confirm the push succeeds, per root `CLAUDE.md` Phase 3.

## Notes for whoever (or whichever session) picks this up
- Original ask was for the manifest.xlsx four columns only — no bin column,
  no fine-grained resin codes, coarse material vocabulary only.
- Roger confirmed (this session) that `training set photos/` was in fact the
  public Kaggle-style "Garbage Classification" dataset, not photos taken by
  the founders, and approved processing all 2,527 images anyway, trusting
  folder name as material where the folder name is a valid material value.
