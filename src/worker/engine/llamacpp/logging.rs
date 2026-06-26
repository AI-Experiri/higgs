//! llama.cpp engine log control — the single home for THIS engine's worker-side
//! logging. Shows up in the higgs "Worker" developer console.
//!
//! Responsibilities:
//! - Install the worker's `tracing` subscriber (stderr, no ANSI — the supervisor
//!   drains stderr as plain text).
//! - Route llama.cpp/ggml's FFI logs into `tracing` (target `"llama-cpp-2"`).
//! - Filter that engine output: a LEVEL gate (INFO+ normal, DEBUG+ verbose) plus
//!   a MODULE gate that hides llama.cpp's unconditional load-time noise (the
//!   per-KV metadata dump and the hyperparameter block) in normal mode.
//! - Flip verbosity live via [`set_engine_verbose`].
//!
//! A different engine (e.g. MLX) ships its own `logging` module with whatever
//! native scheme it uses; nothing here is shared across engines.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use tracing::field::{Field, Visit};
use tracing::{Event, Metadata};
use tracing_subscriber::filter::filter_fn;
use tracing_subscriber::layer::{Context, Filter, SubscriberExt};
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

/// Tracing target the `llama-cpp-2` binding tags all engine (llama.cpp + ggml)
/// events with. Non-engine events (the worker's own future tracing) use other
/// targets and bypass the engine gates entirely.
const ENGINE_TARGET: &str = "llama-cpp-2";

/// Engine modules (the binding's structured `module` field) whose output is pure
/// load-time noise: the per-KV metadata dump (`llama_model_loader`) and the
/// hyperparameter block (`print_info`). llama.cpp emits both unconditionally at
/// INFO with no native gate (upstream's own `// TODO: make optional`), so they
/// are suppressed in normal mode and shown only when verbose. Matched on the
/// structured `module` field value — never on message text.
const NOISY_ENGINE_MODULES: &[&str] = &["llama.cpp::llama_model_loader", "llama.cpp::print_info"];

/// Runtime verbose flag for the engine-log filter. Seeded at spawn from
/// `HIGGS_WORKER_VERBOSE`, flipped live by [`set_engine_verbose`]. Read per log
/// event by the filter, so a toggle takes effect without a worker restart.
static ENGINE_VERBOSE: OnceLock<Arc<AtomicBool>> = OnceLock::new();

/// Engine (llama.cpp/ggml) ERROR lines captured during the current load window.
///
/// A failed `LlamaModel::load_from_file` returns only an opaque binding error
/// ("null result from llama cpp") — the ACTUAL cause (e.g. `unknown model
/// architecture: 'gemma4'`) is emitted by llama.cpp as a SEPARATE log event,
/// decoupled from the FFI `Result`. This buffer taps that stream so a load
/// failure can surface the engine's own diagnostic verbatim: EVERY captured
/// ERROR line, in emission order, with no heuristic guess at which one is the
/// root cause (llama.cpp emits the specific reason first, then a generic
/// `failed to load model` — keeping both is more robust than picking one).
///
/// [`clear_engine_diagnostics`] resets it before each load; the engine's
/// `load` drains it with [`take_engine_diagnostics`] on failure. Bounded by
/// [`MAX_ENGINE_DIAGNOSTICS`] so a pathological engine can't grow it without
/// limit — the first lines carry the root cause, so excess is dropped.
static ENGINE_DIAGNOSTICS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Cap on retained engine-diagnostic lines per load window. Real load failures
/// emit a handful; this is a generous safety bound, not a tuning knob.
const MAX_ENGINE_DIAGNOSTICS: usize = 64;

/// Reset the engine-diagnostic buffer. MUST be called immediately before a load
/// so a prior load's ERROR lines cannot leak into this load's failure reason.
pub fn clear_engine_diagnostics() {
    if let Ok(mut buf) = ENGINE_DIAGNOSTICS.lock() {
        buf.clear();
    }
}

/// Drain the engine ERROR lines captured since the last
/// [`clear_engine_diagnostics`], in emission order. Returns empty when the
/// engine logged nothing (e.g. an OOM kill that printed no line) — callers fall
/// back to the binding's own error string in that case.
pub fn take_engine_diagnostics() -> Vec<String> {
    ENGINE_DIAGNOSTICS
        .lock()
        .map(|mut buf| std::mem::take(&mut *buf))
        .unwrap_or_default()
}

/// Append one engine ERROR line, honoring the [`MAX_ENGINE_DIAGNOSTICS`] bound.
fn record_engine_diagnostic(line: String) {
    if let Ok(mut buf) = ENGINE_DIAGNOSTICS.lock() {
        if buf.len() < MAX_ENGINE_DIAGNOSTICS {
            buf.push(line);
        }
    }
}

/// Install the worker's `tracing` subscriber for llama.cpp engine logs.
///
/// Builds an stderr fmt layer (no ANSI — the supervisor renders the drain as
/// plain text in the UI), gated by [`EngineLogFilter`], then routes the binding's
/// FFI logs into `tracing`. Idempotent per process. Called once at worker start.
pub fn install_worker_logging() {
    let verbose = Arc::new(AtomicBool::new(
        std::env::var("HIGGS_WORKER_VERBOSE").as_deref() == Ok("1"),
    ));
    let _ = ENGINE_VERBOSE.set(verbose.clone());

    let _ = tracing_subscriber::registry()
        // Capture engine ERROR lines for load-failure diagnostics, independent of
        // the fmt layer's verbosity filter. The per-layer filter scopes this layer's
        // callsite INTEREST to ENGINE_TARGET ERROR only — without it an unfiltered
        // layer reports `Interest::always()` for every engine callsite, which would
        // re-enable the DEBUG/INFO engine traffic the fmt layer's level gate
        // suppresses at the source (the binding checks `dispatcher.enabled` before
        // forwarding a native log), defeating normal-mode verbosity.
        .with(EngineDiagnosticCapture.with_filter(filter_fn(|meta| {
            meta.target() == ENGINE_TARGET && *meta.level() == tracing::Level::ERROR
        })))
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                // stderr is captured by the supervisor drain and rendered as plain
                // text in the UI log pane — ANSI color escapes would show as literal
                // `\x1b[..m` garbage, so emit uncolored lines.
                .with_ansi(false)
                .with_filter(EngineLogFilter { verbose }),
        )
        .try_init();

    route_engine_logs_to_tracing();
}

/// Flip the engine-log verbosity at runtime (called by the worker's
/// `higgs/log_level` RPC). No-op if logging was never installed.
pub fn set_engine_verbose(v: bool) {
    if let Some(flag) = ENGINE_VERBOSE.get() {
        flag.store(v, Ordering::Relaxed);
    }
}

/// Route llama.cpp + ggml logs through `tracing` (target [`ENGINE_TARGET`],
/// tagged with the real level + a `module` field) instead of raw-printing every
/// line to stderr at INFO. This is the ONLY place allowed to touch the binding's
/// log hook. Idempotent (the binding installs the callback once).
fn route_engine_logs_to_tracing() {
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
}

/// Per-layer filter for the worker's engine logs. Two gates, both keyed off the
/// live `verbose` flag (flipped by [`set_engine_verbose`]):
/// 1. LEVEL (`enabled`): engine events pass at INFO+ normally, DEBUG+ when
///    verbose. Other targets always pass.
/// 2. MODULE (`event_enabled`): in normal mode, drop the [`NOISY_ENGINE_MODULES`]
///    (KV-dump, hparam block) via the structured `module` field — but always keep
///    engine warnings/errors. Verbose keeps everything.
struct EngineLogFilter {
    verbose: Arc<AtomicBool>,
}

impl EngineLogFilter {
    fn verbose(&self) -> bool {
        self.verbose.load(Ordering::Relaxed)
    }
}

impl<S> Filter<S> for EngineLogFilter {
    fn enabled(&self, meta: &Metadata<'_>, _cx: &Context<'_, S>) -> bool {
        if meta.target() == ENGINE_TARGET {
            let max = if self.verbose() {
                tracing::Level::DEBUG
            } else {
                tracing::Level::INFO
            };
            meta.level() <= &max
        } else {
            true
        }
    }

    fn event_enabled(&self, event: &Event<'_>, _cx: &Context<'_, S>) -> bool {
        // Verbose shows the full engine stream; non-engine events always pass.
        if self.verbose() || event.metadata().target() != ENGINE_TARGET {
            return true;
        }
        // Always surface engine warnings/errors, even from the noisy modules
        // (Level order is ERROR < WARN < INFO, so `<= WARN` is WARN-or-worse).
        if *event.metadata().level() <= tracing::Level::WARN {
            return true;
        }
        let mut visitor = ModuleVisitor::default();
        event.record(&mut visitor);
        match visitor.module {
            Some(module) => !NOISY_ENGINE_MODULES.contains(&module.as_str()),
            None => true,
        }
    }
}

/// Reads the binding's `module` field value off an engine event so the filter can
/// decide whether the event belongs to a [`NOISY_ENGINE_MODULES`] entry.
#[derive(Default)]
struct ModuleVisitor {
    module: Option<String>,
}

impl Visit for ModuleVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "module" {
            self.module = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // Fallback if the value arrives via Debug (a String wraps in quotes).
        if field.name() == "module" && self.module.is_none() {
            self.module = Some(format!("{value:?}").trim_matches('"').to_string());
        }
    }
}

/// Taps engine (llama.cpp/ggml) ERROR events into [`ENGINE_DIAGNOSTICS`] so a
/// load failure can report the engine's own words. A standalone [`Layer`] (not a
/// filter on the fmt layer) so capture is INDEPENDENT of the UI verbosity gate —
/// only the [`ENGINE_TARGET`], only ERROR level (the actual failure cause; WARNs
/// are advisory and stay in the log pane, out of the failure reason).
struct EngineDiagnosticCapture;

impl<S: tracing::Subscriber> Layer<S> for EngineDiagnosticCapture {
    fn on_event(&self, event: &Event<'_>, _cx: Context<'_, S>) {
        let meta = event.metadata();
        // Level order is ERROR < WARN < INFO; ERROR is the single most-severe level,
        // so an exact match isolates engine errors from advisory warnings.
        if meta.target() != ENGINE_TARGET || *meta.level() != tracing::Level::ERROR {
            return;
        }
        let mut visitor = MessageVisitor::default();
        event.record(&mut visitor);
        if let Some(msg) = visitor.message {
            record_engine_diagnostic(msg);
        }
    }
}

/// Reads the `message` field off an engine event so [`EngineDiagnosticCapture`]
/// can retain the engine's own failure text. The binding renders the line via
/// `Debug` (format args); `record_str` is a defensive fallback.
#[derive(Default)]
struct MessageVisitor {
    message: Option<String>,
}

impl Visit for MessageVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "message" {
            self.message = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        // The `message` field arrives as format args, whose `Debug` writes the
        // rendered text directly (no surrounding quotes to strip).
        if field.name() == "message" && self.message.is_none() {
            self.message = Some(format!("{value:?}"));
        }
    }
}

#[cfg(test)]
#[path = "logging_tests.rs"]
mod tests;
