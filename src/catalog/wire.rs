//! Wire types for the model-search catalog — the ts-rs surface jigglebot's
//! Model Search UI consumes (and the CLI renders). Types only: assembly lives
//! in [`crate::catalog::service`], Hub transport in [`crate::catalog::source`].

use crate::system::FitAssessment;

higgs_const_enum! {
    /// Sort order for a catalog search. The source maps each variant onto the
    /// Hub API's sort key (`downloads` / `likes` / `lastModified` /
    /// `trendingScore`); the wire never carries a raw Hub sort string.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum CatalogSort {
        /// Most-downloaded first — the default.
        #[default]
        Downloads,
        /// Most-liked first.
        Likes,
        /// Most-recently-updated first.
        Updated,
        /// The Hub's trending score.
        Trending,
    }
}

higgs_ts! {
    /// One search against the Hub model catalog. Sent by jigglebot's Model
    /// Search tab (and built by the CLI). Every field except `search` has a
    /// serde default so a caller can send just the query text.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    pub struct CatalogQuery {
        /// Free-text query (matches repo ids and descriptions on the Hub).
        pub search: String,
        /// Restrict to one author/organization (used for "more by publisher").
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub author: Option<String>,
        /// Result ordering; absent = [`CatalogSort::Downloads`].
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub sort: Option<CatalogSort>,
        /// Maximum rows to return, clamped to [`MAX_SEARCH_LIMIT`]; absent (or
        /// `0`) = the default page size.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub limit: Option<u32>,
        /// `true` = drop rows whose smallest shipped quant is ESTIMATED not to
        /// fit this machine (rows without enough data for a verdict are kept —
        /// unknown is never treated as incompatible). Absent/`false` = all rows.
        /// NB the filtered set is best-effort, not deterministic click-to-click:
        /// a row whose enrichment times out on one search is kept as unknown,
        /// and may enrich (and drop) on the next.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub compatible_only: Option<bool>,
    }
}

/// Hard cap on catalog search results per query — one UI page, not a crawl.
pub const MAX_SEARCH_LIMIT: u32 = 50;

/// Default catalog page size when the caller sends `limit: 0` (or omits it).
pub const DEFAULT_SEARCH_LIMIT: u32 = 25;

higgs_ts! {
    /// The Hub's GGUF metadata block for a repo (`gguf` on the model-info
    /// response) reduced to the fields the UI badges: architecture, total
    /// parameter count, and training context length. Present only on repos
    /// where the Hub reports it (detail responses; search rows omit it).
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct CatalogGgufMeta {
        /// Model architecture (e.g. `"llama"`, `"qwen3"`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub arch: Option<String>,
        /// Total parameter count (e.g. `8030261248` for an 8B model).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub params_total: Option<u64>,
        /// Training context length in tokens.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub ctx_train: Option<u64>,
    }
}

higgs_ts! {
    /// One catalog search row: a Hub GGUF model repo plus whether any quant of
    /// it is already on this machine.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct CatalogModelSummary {
        /// Hub repo id (`org/model`) — also the higgs model id after download.
        pub id: String,
        /// Repo author/organization.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub author: Option<String>,
        /// All-time download count on the Hub.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub downloads: Option<u64>,
        /// Like count on the Hub.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub likes: Option<u64>,
        /// Last-modified timestamp (ISO-8601, verbatim from the Hub).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub updated: Option<String>,
        /// The Hub pipeline tag (e.g. `"text-generation"`).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub pipeline: Option<String>,
        /// GGUF metadata badge fields, when the Hub response carried them.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub gguf: Option<CatalogGgufMeta>,
        /// Advisory repo-level fit verdict: the ESTIMATED footprint of the
        /// smallest quant the repo ships (parameter count × the quant family's
        /// effective bytes-per-weight) against this machine's VRAM headroom.
        /// `None` when the estimate has no inputs (no parameter count, no
        /// labeled quant, or no reported VRAM) — never a fake verdict. Detail
        /// responses replace the estimate with the real smallest-file size.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub fit: Option<FitAssessment>,
        /// `true` when the local scan already has a model under this repo id.
        pub downloaded: bool,
    }
}

higgs_ts! {
    /// Catalog search response: the mapped rows, newest query wins in the UI.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct CatalogSearchResponse {
        /// Search rows in the requested sort order.
        pub models: Vec<CatalogModelSummary>,
    }
}

higgs_ts! {
    /// One downloadable GGUF file of a repo — the quant-picker row.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct CatalogQuant {
        /// File name inside the repo (e.g. `"model-Q4_K_M.gguf"`).
        pub file: String,
        /// Quantization label parsed from the file name (e.g. `"Q4_K_M"`),
        /// when the name follows the common convention.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub quant: Option<String>,
        /// File size in bytes, when the Hub reported it.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub size_bytes: Option<u64>,
        /// `true` when this exact file is already on this machine.
        pub downloaded: bool,
        /// Size-based fit verdict against this machine's VRAM headroom
        /// (advisory, pre-download — the file size is a lower bound for the
        /// resident footprint). `None` when the size or hardware is unknown.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub fit: Option<FitAssessment>,
    }
}

higgs_ts! {
    /// Full catalog detail for one repo — the right-hand detail pane.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct CatalogModelDetail {
        /// The repo's search-row fields (with `gguf` populated when reported).
        pub summary: CatalogModelSummary,
        /// Hub tags (capability/library badges are derived from these).
        pub tags: Vec<String>,
        /// The repo README (markdown), truncated to a bounded size; `None`
        /// when the repo has none or the fetch failed (best-effort).
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub readme: Option<String>,
        /// Every `.gguf` file in the repo, one quant-picker row each.
        pub quants: Vec<CatalogQuant>,
        /// The file the download picker should preselect — higgs's
        /// [`default_quant`](crate::catalog::service::default_quant) policy
        /// (`Q4_K_M` → largest that fits → smallest). `None` when the repo
        /// ships no downloadable quant.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub default_file: Option<String>,
        /// Other repos by the same author (self excluded) — "more by publisher".
        pub more_by_author: Vec<CatalogModelSummary>,
    }
}

higgs_const_enum! {
    /// Lifecycle phase of a catalog download, pushed as a live
    /// [`ModelDownloadEvent`] over the download-event subscription
    /// ([`Higgs::subscribe_download_events`](crate::api::Higgs::subscribe_download_events)).
    /// `Starting`/`Downloading` are progress phases; `Done`/`Failed`/
    /// `Cancelled` are terminal — a row that saw any of the three never
    /// gets another event for the same attempt.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ModelDownloadPhase {
        /// The download was accepted and the transfer is being opened.
        Starting,
        /// Bytes are flowing; `downloaded_bytes`/`total_bytes` carry progress.
        Downloading,
        /// Terminal: the file is on disk at `path`.
        Done,
        /// Terminal: the download failed; `code` carries the `HGxxx`.
        Failed,
        /// Terminal for THIS ATTEMPT, discriminated by `code`:
        /// `HG089` = a real cancel (operator action / caller drop) — the
        /// transfer stopped, nothing landed (its own temp guard cleaned
        /// up, failures visible only in tracing); `HG090` = this attempt
        /// yielded to a transfer that CONTINUES elsewhere (duplicate
        /// refusal, or a hub-side drop of a node transfer that survives by
        /// design) — progress fields may carry the live transfer's bytes.
        /// Both render as neutral/info states, never an error toast.
        Cancelled,
    }
}

higgs_ts! {
    /// A live catalog-download progress event, delivered over
    /// [`Higgs::subscribe_download_events`](crate::api::Higgs::subscribe_download_events).
    /// Progress events are throttled by the emitter; terminal phases always
    /// emit.
    #[derive(Debug, Clone, serde::Serialize)]
    pub struct ModelDownloadEvent {
        /// Fleet node the download runs ON (`Higgs::model_download_on`);
        /// absent = this machine (`Higgs::model_download`). One event stream
        /// serves every target — subscribers key progress by
        /// `(node, repo, file)`. A duplicate re-issue for a key that is
        /// already transferring TERMINALIZES this attempt with
        /// `Cancelled` carrying `code: "HG090"` — the ORIGINAL transfer
        /// keeps flowing its own event stream (Downloading→Done); the
        /// row's ongoing truth is `NodeView.downloads`, not this refused
        /// attempt's event stream. Every HG090 producer (hub-facade
        /// pre-Starting refusal, node-side post-Starting refusal, LOCAL
        /// cross-process adopt, and the wire-attested hub-side drop of a
        /// node transfer that survives by design) shares this
        /// Cancelled-terminal shape — the `HG090` code on the terminal is
        /// what tags the UI to render it as an info state ("the transfer
        /// continues elsewhere"), never an error.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub node: Option<String>,
        /// Hub repo id (`org/model`) being downloaded.
        pub repo: String,
        /// File name inside the repo.
        pub file: String,
        /// The phase this event announces.
        pub phase: ModelDownloadPhase,
        /// Bytes received so far.
        #[ts(type = "number")]
        pub downloaded_bytes: u64,
        /// Total bytes, when the transfer reported a length.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub total_bytes: Option<u64>,
        /// Unix-ms when the event was emitted.
        #[ts(type = "number")]
        pub at_ms: u64,
        /// The `HGxxx` diagnostic code. Present on
        /// [`ModelDownloadPhase::Failed`] (what failed) and on
        /// [`ModelDownloadPhase::Cancelled`] — the two codes that ride
        /// Cancelled are `HG089` (the transfer STOPPED: a local caller
        /// drop, an unattested remote drop, or — once the cancel-dispatch
        /// slice ships — an operator cancel; nothing landed) and `HG090`
        /// (this attempt yielded to a transfer that CONTINUES elsewhere:
        /// duplicate refusal, cross-process adopt, or a wire-attested
        /// hub-side drop; the live transfer's row continues via
        /// `NodeView.downloads`). UI classification: `HG089`/`HG090`
        /// render as info states, never as error toasts.
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub code: Option<String>,
        /// Final on-disk path — present only on [`ModelDownloadPhase::Done`].
        #[serde(skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub path: Option<String>,
    }
}

higgs_const_enum! {
    /// Lifecycle status of one entry in the machine-local downloads LEDGER
    /// (`~/.higgs/models/.downloads.json`, [`crate::catalog::ledger`]): the on-disk,
    /// cross-process record of what this machine is downloading and has
    /// downloaded. Distinct from [`ModelDownloadPhase`] (the live PUSH event
    /// stream): the ledger is the referable state, the events are its motion.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum DownloadLedgerStatus {
        /// The transfer is in flight in the process named by `pid`.
        Downloading,
        /// The file landed whole at `path`.
        Done,
        /// The transfer ended without landing the file (`detail` says why —
        /// including "downloader process exited", the dead-pid sweep's verdict
        /// on a crashed downloader's stale entry).
        Failed,
        /// The transfer was cancelled mid-flight. Temp cleanup is done by
        /// the transfer's OWN per-attempt drop guard (never a blanket
        /// sweep); an unlink failure is visible only in tracing — this
        /// status records the outcome, not a cleanup receipt.
        Cancelled,
    }
}

higgs_ts! {
    /// One machine-local downloads-ledger entry ([`crate::catalog::ledger`]):
    /// a live transfer (any process on this machine — node daemon, embedded
    /// hub, `higgs download` CLI) or a terminal history record. `(repo, file)`
    /// is the download identity (same `dest_path` rules as everywhere);
    /// `pid` names the owning process while `status` is `downloading`.
    #[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
    pub struct DownloadLedgerEntry {
        pub repo: String,
        pub file: String,
        /// Owning process while downloading (dead-pid entries are swept).
        pub pid: u32,
        /// That process's kernel start time (platform units) — the pid-reuse
        /// disambiguator: liveness requires the pid AND its start time to
        /// match, so a recycled pid never masquerades as a live transfer.
        /// Absent on legacy entries / unqueryable platforms (pid-only then).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub pid_started_at: Option<u64>,
        #[ts(type = "number")]
        pub started_at_ms: u64,
        #[ts(type = "number")]
        pub downloaded: u64,
        /// Absent when the server sent no content length.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub total: Option<u64>,
        pub status: DownloadLedgerStatus,
        /// Set on every terminal status.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional, type = "number")]
        pub ended_at_ms: Option<u64>,
        /// Final on-disk path — [`DownloadLedgerStatus::Done`] only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub path: Option<String>,
        /// Failure reason — [`DownloadLedgerStatus::Failed`] only.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[ts(optional)]
        pub detail: Option<String>,
    }
}
