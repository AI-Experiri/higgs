# `catalog/` — Hugging Face model search, detail, and download

The model-search catalog: higgs discovers GGUF model repos on the **Hugging
Face Hub** (the ONLY source), inspects one repo's quants/README/fit, and
downloads a chosen quant into `~/.higgs/models/` — the LM-Studio-layout dir
that `HiggsConfig::default()` includes as a scan root, so a just-downloaded
model shows up in the very next scan everywhere (embedder, node, CLI).

Every Hub touch goes through the pinned `huggingface-hub` crate (the same
`HFClient` + `HIGGS_HF_ENDPOINT` override [`../hub.rs`](../hub.rs) uses);
nothing here hand-rolls Hub HTTP, search, or caching. Ops are **on-click
only** — no cache, no timers, no background refresh.

An **empty query is the browse page** — the Hub's full GGUF listing in the
requested sort order (downloads/likes/updated/trending). Search rows are
**enriched** with their repo's `gguf` block (the list endpoint omits it) via
bounded, order-preserving per-row info fetches that degrade to the bare row;
from it each row gets an **advisory fit estimate** (parameter count × the
cheapest shipped quant family's effective bytes-per-weight, vs the VRAM
headroom budget). `CatalogQuery.compatible_only` drops rows whose estimate
says "does not fit" — rows with no estimate are never treated as misfits. The
detail response replaces the estimate with the real smallest-file verdict and
names higgs's `default_file` pick for the download picker.

Consumers:

- **Embed facade** — `Higgs::model_search` / `model_detail` / `model_download`
  (+ `subscribe_download_events`) in [`../api/embed.rs`](../api/embed.rs), the
  ops jigglebot's Model Search tab drives.
- **CLI** — `higgs model <search|show|download>` (`cli.rs`).

## File map

| File | Responsibility |
|------|----------------|
| `mod.rs` | export barrel + module docs (no logic) |
| `wire.rs` | ts-rs wire types (`CatalogQuery`, `CatalogModelSummary`, `CatalogModelDetail`, `CatalogQuant`, `ModelDownloadEvent`, …) |
| `source.rs` | the `CatalogSource` seam + the ONE production impl `HfSource` on the crate's `HFClient`; bounded README fetch |
| `service.rs` | pure assembly: summaries, gguf badge, quant rows (sizes/labels/fit/downloaded), default-quant pick, `LocalInventory`; async `search`/`detail` orchestration |
| `pull.rs` | one download entry over `crate::download::download_dual` (hub primary + reqwest fallback, atomic write) + the progress-event `ProgressGate` |
| `cli.rs` | `higgs model` parsing (pure), rendering (pure), and the runtime driver |

Unit tests live in `*_tests.rs` siblings; the end-to-end path (facade + CLI
against a loopback fixture Hub) is `tests/catalog.rs`.
