//! Single source of truth for where higgs's ts-rs types emit their TypeScript.
//!
//! ts-rs's `#[ts(export_to = …)]` requires a string LITERAL at derive time, so
//! the path cannot be a `const`. Instead, [`higgs_ts!`] wraps each ts-rs type
//! and injects the export attribute — keeping the path in exactly ONE place.
//! Change the literal in the macro below to relocate every higgs `.ts` file.
//!
//! All higgs generated types land in the crate-local `bindings/higgs/` dir.
//! higgs is a standalone crate, so it emits its TypeScript into its own tree;
//! embedders (e.g. jigglebot) consume these as published/vendored types rather
//! than higgs reaching into the embedder's frontend.

/// Wrap a higgs ts-rs type so it derives [`ts_rs::TS`] and exports to the single
/// higgs generated dir. Apply to every higgs type that crosses to the frontend.
///
/// All other derives (`Debug`/`Clone`/`serde::*`) and field-level `#[ts(…)]` /
/// `#[serde(…)]` attributes are written as usual — they pass through unchanged.
/// Do NOT add `#[derive(ts_rs::TS)]` or `#[ts(export, …)]` yourself; the macro
/// owns both.
macro_rules! higgs_ts {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[derive(ts_rs::TS)]
        #[ts(export, export_to = "higgs/")]
        $vis struct $name { $($body)* }
    };
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[derive(ts_rs::TS)]
        #[ts(export, export_to = "higgs/")]
        $vis enum $name { $($body)* }
    };
}

/// Like [`higgs_ts!`] but for **unit-variant** enums that should emit a TypeScript
/// const-OBJECT (usable as a VALUE — `Foo.Bar`) instead of a ts-rs `"a" | "b"`
/// union — via [`higgs_macros::TsConstEnum`], consistent with jigglebot's enum
/// bindings. Apply to plain C-style enums; a variant-with-fields enum stays on
/// [`higgs_ts!`] (it needs ts-rs's discriminated-union output).
///
/// Keeps `ts_rs::TS` + `export_to` (so ts-rs resolves the type + import path for
/// dependent structs) but NOT `#[ts(export)]`: that would make ts-rs write a
/// competing union file racing TsConstEnum's writer. TsConstEnum emits a
/// `#[ignore]`d `macro_run_*` test that writes the .ts; regenerate it via
/// `cargo test macro_run -- --ignored` (wired into `scripts/quality.sh`).
macro_rules! higgs_const_enum {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident { $($body:tt)* }
    ) => {
        $(#[$meta])*
        #[derive(ts_rs::TS, higgs_macros::TsConstEnum)]
        #[ts(export_to = "higgs/")]
        $vis enum $name { $($body)* }
    };
}
