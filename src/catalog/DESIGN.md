# `catalog/` — design notes

## One source, one crate, on-click only

The catalog is a **thin, honest view over the Hugging Face Hub API** — the only
model source for now — reached exclusively through the pinned
`huggingface-hub` crate (`list_models`, repo `info`, `get_paths_info`,
`download_file_stream`). No hand-rolled Hub HTTP, no local catalog store, no
staleness to manage: every search/detail/download happens on the user's click
and reflects the Hub at that instant. `HIGGS_HF_ENDPOINT` redirects everything
(mirror, proxy, test fixture) exactly like the rest of higgs's Hub traffic.

## Seams (why each piece is testable)

```
 CatalogQuery ──▶ service::search ──▶ CatalogSearchResponse
 repo id      ──▶ service::detail ──▶ CatalogModelDetail
                      │
                      ▼ (trait CatalogSource — HfSource in production,
                         a canned fake in service_tests)
                huggingface-hub crate ──▶ Hub API
```

- **`CatalogSource`** is the network seam: `service` never sees the crate's
  client, only its `ModelInfo` values, so assembly logic is unit-tested with
  fixture JSON. `HfSource` is deliberately thin — parameter mapping is in pure
  fns (`hub_sort`, `effective_limit`, `sizes_from_entries`, `clip_utf8`) that
  have their own tests; the async methods are classified pass-throughs.
- **`LocalInventory`** is the disk seam: downloaded-state is computed against a
  snapshot (facade: one `scan()`; CLI: a walk of the LM-Studio-layout roots),
  never per-row I/O.
- **`pull`** delegates every hard property (wire validation, path escapes,
  atomic `.part` writes, hub-primary/reqwest-fallback) to `crate::download`,
  which already proved them; `pull_with` injects fetchers for tests.

## Detail assembly degrades per-part

Only the repo info itself is load-bearing. Sizes (`paths-info`), the README,
and the more-by-author rows each degrade independently on failure OR stall
(each is bounded by `ENRICHMENT_TIMEOUT` — the pinned client has no request
timeout, so a black-holed connection must not hold the pane) — one Hub hiccup
never blanks or hangs the detail. The README fetch is byte-bounded
(`README_MAX_BYTES`, stream aborted at the cap) so a pathological repo cannot
balloon a response.

## Quant rows are the downloadable set

A row the picker offers must be a row `pull` can land: only siblings that are
a single safe `*.gguf` file name (the scanned `<org>/<model>/<file>.gguf`
layout, `download::is_safe_segment`) become quant rows. A repo shipping GGUFs
in subdirectories has those files omitted rather than shown as a
guaranteed-to-fail download (a deliberate limitation, matching the scan
layout).

## Fit verdicts never guess

A quant row gets a `FitAssessment` (the existing `system::fits_vram` with the
canonical headroom fraction) only when BOTH the file size and the machine's
VRAM total are known. Unknown size or no device sample → `fit: None`, shown as
"unknown", never a fabricated verdict. The file size is an honest lower bound
for the resident footprint (weights only — KV/compute come on top), which is
exactly what an advisory pre-download badge can claim.

## Browse + the row-level estimate (search only)

An empty `CatalogQuery.search` is the browse page — the Hub's GGUF listing in
the requested sort order, the default listing the UI opens on. Because the
list endpoint returns no `gguf` block, `search` enriches each row with a
bounded per-row `info` fetch (`SEARCH_ENRICH_CONCURRENCY` at a time, each
under `ENRICHMENT_TIMEOUT`, order preserved, degrade-to-bare-row on failure).
Rows have no file sizes, so the row-level `fit` is an explicitly ADVISORY
estimate: parameter count × the cheapest shipped quant family's effective
bytes-per-weight (`quant_bytes_per_param` — whole-file GGUF sizes over
parameter count, embed/output tensors included). It exists to answer "could
this machine plausibly run ANY quant of this repo"; the detail pane replaces
it with the real smallest-file verdict as soon as sizes are known.
`compatible_only` drops only PROVEN misfits (`fits == false`); a row with no
estimate is never treated as one — absence of data is not a verdict. The Hub
is over-asked (`MAX_SEARCH_LIMIT`) when filtering so the page stays full, then
truncated to the requested limit.

Rows whose estimate has NO inputs fall back to REAL sizes in the same bounded
pass (`real_min_size`): a known sibling size when the Hub reported one, else
one `paths-info` fetch — so "no metadata" degrades to an honest real-size
verdict rather than a blind spot. Accepted residuals of this design: one page
mixes estimate-based and real-size verdicts (inherent to the fallback), and
the bpp table is per quant FAMILY, so same-digit variants (`Q4_K_S` vs
`Q4_K_M`) share one number, and it is calibrated to DENSE models (large MoE
repos carry proportionally less embed/output overhead, so their true bpp
runs a little lower) — a ±10–15% error bar on an advisory field the detail
pane replaces with real sizes. The fallback runs only when VRAM is known
(no verdict is possible otherwise). The filtered set remains best-effort, not
deterministic click-to-click: an enrichment that times out on one search may
resolve on the next (documented on the wire field).

## The default download pick is higgs's

`CatalogModelDetail.default_file` names the quant the picker preselects —
`default_quant`'s policy (`Q4_K_M` → largest that fits → smallest) computed
server-side so every consumer (UI picker, CLI no-file download) preselects
the SAME file for the same machine; the UI never re-derives it.

## Download progress is push, not poll

`Higgs::model_download` emits `ModelDownloadEvent` over a broadcast channel
(`subscribe_download_events`), mirroring the model-load events: `Starting` →
`Downloading` (throttled to one per 250ms by `ProgressGate`) → terminal
`Done`/`Failed` (with the `HGxxx` code). One download per `(repo, file)` is in
flight at a time — a concurrent duplicate is refused (HG025, before any event)
so subscribers never see two interleaved streams for the same file and a
multi-GB transfer never runs twice. The CLI prints its own stderr bar from
the same `pull` progress callback. Failures reuse the existing `HG029`–`HG036`
hub classification — no new codes.

## Errors

`classify_hf` (in `../hub.rs`) maps every crate error to the established
distinct codes: auth (`HG029`), not-found (`HG030`), rate-limit (`HG031`),
HTTP status (`HG032`), transport (`HG033`), client (`HG035`); the dual-path
download terminal is `HG036`. A repo with no README is `Ok(None)`, not an
error.
