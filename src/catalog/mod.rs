//! Model-search catalog — Hugging Face-backed model discovery for higgs.
//!
//! higgs establishes a catalog over the Hugging Face Hub (the ONLY source):
//! search the Hub for GGUF model repos, inspect one repo's quants/README/fit,
//! and download a chosen quant into `~/.higgs/models/`. Every Hub touch goes
//! through the pinned `huggingface-hub` crate (the same client `crate::hub`
//! uses) — nothing here hand-rolls Hub HTTP.
//!
//! Components, kept separate:
//! - [`wire`] — the ts-rs wire types jigglebot's Model Search UI consumes.
//! - [`source`] — the [`source::CatalogSource`] seam + the production
//!   [`source::HfSource`] on the crate's `HFClient`.
//! - [`service`] — pure assembly from Hub responses to wire types (fit,
//!   quant labels, downloaded-state, default-quant pick).
//! - [`pull`] — one download entry over the existing dual-path downloader.
//! - [`cli`] — `higgs model <search|show|download>`.
//!
//! The facade ops (`Higgs::model_search`/`model_detail`/`model_download`) in
//! `crate::api::embed` are thin delegations into this module.

pub mod cli;
pub mod pull;
pub mod service;
pub mod source;
#[cfg(test)]
pub(crate) mod test_support;
pub mod wire;

pub use service::LocalInventory;
pub use source::{CatalogSource, HfSource};
pub use wire::{
    CatalogGgufMeta, CatalogModelDetail, CatalogModelSummary, CatalogQuant, CatalogQuery,
    CatalogSearchResponse, CatalogSort, ModelDownloadEvent, ModelDownloadPhase,
};
