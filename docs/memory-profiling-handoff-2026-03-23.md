# OpenRecall Memory Profiling Handoff (2026-03-23)

## Scope

This document captures all memory/performance experiments run in this session, including:

- Full-app RSS/PSS tracking over time
- Parent vs child (`ffmpeg`) memory decomposition
- Anonymous mapping attribution (`smaps`)
- OCR and embedding subsystem isolation tests
- `MALLOC_ARENA_MAX` A/B tests on full app
- AV1 preset and encoder tradeoff tests
- Runtime check for PyTorch (`torch`) residency

All measurements were taken on Linux in this workspace.

---

## Key Findings (TL;DR)

1. The observed memory growth to a plateau is real (not just illusion): startup low RSS rises as native allocators warm up and retain memory.
2. Largest component is parent-process anonymous memory (mostly OCR/runtime allocator pools), not file-backed mappings.
3. `ffmpeg` children are substantial (~20% of total in a representative sample), but not the majority.
4. `torch` is not loaded in the current runtime process and is not the primary source of memory in this setup.
5. `MALLOC_ARENA_MAX=2` helps in some runs; `=1` can reduce average/end RSS further but is more volatile.

---

## Code/Config Changes Applied During Session

- OCR CPU memory arena default set to disabled in `openrecall/ocr.py`.
- OCR CPU thread default set to `2` in `openrecall/ocr.py`.
- AV1 preset default changed to `9` in `openrecall/config.py`.
- `MALLOC_ARENA_MAX` startup default set to `2` (when unset) in `openrecall/app.py`.
- README updated to reflect relevant defaults.

---

## Experiment Log

## 1) Full app memory timeline (early run, 1s interval)

Run showed parent process around ~0.93–0.94 GB RSS with transient child spikes.

Example sampled rows:

- `t=0s`: RSS `928,228 KB`, PSS `918,086 KB`, anon `868,824 KB`
- `t=15s`: RSS `941,968 KB`, PSS `931,665 KB`, anon `892,636 KB`, child RSS `34,964 KB`
- Later spikes: child RSS observed up to `312,828 KB`

Note: this run likely underrepresented steady state due to atypical conditions (1s capture and concurrent app activity).

---

## 2) Full app decomposition at high-memory state (representative)

Representative `smaps_rollup` (parent process):

- RSS: `1,411,712 KB`
- PSS: `1,402,717 KB`
- PSS_Anon: `1,390,416 KB`
- PSS_File: `12,301 KB`
- Anonymous: `1,390,416 KB`
- AnonHugePages: `225,280 KB`
- Swap: `271,440 KB`

At one point with active children:

- Parent RSS: `1,190,364 KB`
- Child RSS (2 x `ffmpeg`): `172,036 KB` + `135,864 KB` = `307,900 KB`
- Total sampled RSS: `1,498,264 KB`

Fractions from this sample:

- Parent total: `79.4%`
- Children total: `20.6%`
- Parent anonymous portion (relative to total): `73.9%`
- Parent file-backed portion (relative to total): `4.8%`

---

## 3) Anonymous mapping attribution (`smaps`)

Path-level top contributors at peak-like state:

- `[anon]`: PSS `1,398,636 KB`, RSS `1,398,636 KB`, swap `238,732 KB`, AHP `247,808 KB`
- `[heap]`: PSS `4,536 KB`
- Largest shared object mappings were small by comparison.

Anon-region size bins:

- `>=200MB`: 1 region, `482,360 KB`
- `100–200MB`: 0 regions
- `50–100MB`: 1 region, `98,336 KB`
- `20–50MB`: 10 regions, `329,160 KB`
- `10–20MB`: 16 regions, `213,548 KB`
- `<10MB`: 130 regions, `164,572 KB`

Interpretation: memory is dominated by a few large and many medium anonymous arenas/workspaces, consistent with native allocator + ONNX runtime behavior.

---

## 4) OCR subsystem isolation

### 4.1 Arena ON vs OFF memory impact (isolated OCR stress)

At 1080p repeated OCR inference:

- Arena ON (`enable_cpu_mem_arena=True`): RSS stabilized around `~860–886 MB`
- Arena OFF (`False`): RSS around `~395–422 MB`

### 4.2 Arena ON vs OFF latency (isolated)

Average OCR latency (same synthetic input):

- Arena ON: `~713.7 ms`
- Arena OFF: `~627.1 ms`

In this environment, arena OFF reduced memory significantly and did not harm latency in the tested pattern.

### 4.3 OCR thread count tradeoff (arena OFF)

Memory:

- Threads=1: `448,124 KB`
- Threads=2: `432,196 KB`
- Threads=4: `565,632 KB`

Latency:

- Threads=1: avg `1087.7 ms`
- Threads=2: avg `718.3 ms`
- Threads=4: avg `658.5 ms`

Practical compromise selected: default OCR threads `2`.

---

## 5) Embedding subsystem isolation

Standalone embedding process test:

- Start: `10,192 KB`
- After importing `openrecall.nlp`: `39,956 KB`
- After first embedding: `195,644 KB`
- After second embedding: `196,760 KB`

Interpretation: embedding stack costs roughly ~`190–200 MB` in this environment once loaded.

---

## 6) AV1 preset tests (`libsvtav1`, CRF 38)

Synthetic benchmark (1080p @ 2fps, 20s):

Baseline preset 8:

- size `1,059,586 B`
- max RSS `730,520 KB`
- elapsed `0.92 s`

Compared to preset 8:

- Preset 9: size `+5.2%`, RSS `-8.3%`, speed `+13.3%`
- Preset 10: size `+13.3%`, RSS `-10.4%`, speed `+14.9%`
- Preset 11: size `+12.6%`, RSS `-11.4%`, speed `+44.6%` (non-RTC warning)

Default was updated to preset `9` as balanced choice.

---

## 7) Encoder comparison: `libsvtav1` vs `libaom-av1` vs `av1_vaapi`

Same synthetic workload:

- `libsvtav1` (preset 9):
	- elapsed `0.67 s`
	- max RSS `661,324 KB`
	- size `1,114,914 B`
- `libaom-av1` (`cpu-used=8`):
	- elapsed `5.64 s`
	- max RSS `539,336 KB`
	- size `790,228 B`
- `av1_vaapi`:
	- failed on this host (`Function not implemented` / encoder open failure)

Interpretation: `libaom-av1` used less memory but was far slower for this workload; unsuitable for strict low-latency constraints.

---

## 8) Full-app `MALLOC_ARENA_MAX` tests @ capture interval 10s

### Baseline (unset)

- captures: `12`
- RSS start/end: `247,872 KB` -> `952,604 KB`
- RSS peak: `952,604 KB`
- RSS avg: `750,912 KB`
- Child peak: `242,144 KB`

### `MALLOC_ARENA_MAX=2`

- captures: `11`
- RSS start/end: `247,684 KB` -> `709,500 KB`
- RSS peak: `849,788 KB`
- RSS avg: `653,977 KB`
- Child peak: `263,592 KB`

### `MALLOC_ARENA_MAX=1`

- captures: `15`
- RSS start/end: `247,544 KB` -> `580,752 KB`
- RSS peak: `857,760 KB`
- RSS avg: `532,795 KB`
- Child peak: `479,020 KB` (higher volatility)

Interpretation: lower arena cap can reduce average/end RSS but may increase volatility; `2` chosen as safer default.

---

## 9) PyTorch (`torch`) runtime check

Question: Is `torch` loaded and contributing memory?

Findings:

- No direct `torch` imports in current application code paths.
- Live process `/proc/<pid>/maps` and FD checks found no `torch`/`libtorch` mappings/files.
- Therefore, in current runtime, `torch` is not a primary contributor to observed plateau.

Note: some metadata files still list torch-related requirements; these appear stale relative to active runtime path.

---

## Overall Attribution (Practical)

At high steady state with two monitor encoders active:

- ~`20%`: `ffmpeg` child processes (encoding)
- ~`75%`: parent anonymous memory (OCR/runtime allocators, native workspaces, image/array churn)
- ~`5%`: parent file-backed mappings

Within parent process, approximate contributors:

- OCR/runtime: largest share (hundreds of MB)
- Embeddings: ~`190–200 MB` once loaded
- Core app/framework/import baseline: a few hundred MB before full warmup

---

## Caveats

- Some earlier runs were partial/invalid due to process overlap or startup race conditions; those were explicitly discarded in analysis.
- Memory values depend on active UI browsing (`/frame` decode requests), monitor count/resolution, and capture activity.
- Different kernels/drivers/allocators may shift absolute numbers while preserving the same pattern.

---

## Suggested Next Deep-Dive (if needed)

For exact per-stage deltas in a single clean run:

1. Launch one app instance only.
2. Freeze UI interaction.
3. Step through capture count milestones.
4. Record parent `smaps_rollup` + child memory each milestone.
5. Correlate with `recent_timings` (`ocr_ms`, `embedding_ms`, `encode_ms`).

This would produce the tightest possible stage-attribution table.

