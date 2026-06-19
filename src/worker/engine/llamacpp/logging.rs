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
use std::sync::{Arc, OnceLock};

use tracing::field::{Field, Visit};
use tracing::{Event, Metadata};
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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use tracing_subscriber::fmt::MakeWriter;

    use super::*;

    /// Capturing writer: collects everything the fmt layer emits so a test can
    /// assert which engine lines survived the filter.
    #[derive(Clone, Default)]
    struct BufWriter(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for BufWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl<'a> MakeWriter<'a> for BufWriter {
        type Writer = BufWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Emit a representative engine event mix under [`EngineLogFilter`] at the
    /// given verbosity and return everything that was written.
    fn captured(verbose: bool) -> String {
        let buf = BufWriter::default();
        let flag = Arc::new(AtomicBool::new(verbose));
        let subscriber = tracing_subscriber::registry().with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(buf.clone())
                .with_filter(EngineLogFilter { verbose: flag }),
        );
        tracing::subscriber::with_default(subscriber, || {
            // INFO from a noisy module — the KV-dump.
            tracing::info!(target: ENGINE_TARGET, module = "llama.cpp::llama_model_loader", "kv 0 dump");
            // INFO from a noisy module — the hparam block.
            tracing::info!(target: ENGINE_TARGET, module = "llama.cpp::print_info", "arch = bert");
            // INFO from a useful module — must stay in normal mode.
            tracing::info!(target: ENGINE_TARGET, module = "llama.cpp::load_tensors", "offloaded 43/43 layers");
            // WARN from a noisy module — must ALWAYS surface.
            tracing::warn!(target: ENGINE_TARGET, module = "llama.cpp::llama_model_loader", "tokenizer config may be incorrect");
            // DEBUG engine event — level-gated off in normal mode.
            tracing::debug!(target: ENGINE_TARGET, module = "llama_cpp_2::model", "Loaded model");
            // Non-engine event — always passes.
            tracing::info!(target: "higgs", "app event");
        });
        let bytes = buf.0.lock().unwrap().clone();
        String::from_utf8(bytes).unwrap()
    }

    #[test]
    fn normal_mode_hides_kv_dump_and_print_info_keeps_useful() {
        let out = captured(false);
        assert!(
            !out.contains("kv 0 dump"),
            "KV-dump module hidden in normal mode"
        );
        assert!(
            !out.contains("arch = bert"),
            "print_info module hidden in normal mode"
        );
        assert!(
            out.contains("offloaded 43/43 layers"),
            "useful engine INFO stays"
        );
        assert!(
            out.contains("tokenizer config may be incorrect"),
            "engine WARN always surfaces"
        );
        assert!(
            !out.contains("Loaded model"),
            "engine DEBUG level-gated off in normal mode"
        );
        assert!(out.contains("app event"), "non-engine events always pass");
    }

    #[test]
    fn verbose_mode_shows_everything() {
        let out = captured(true);
        assert!(out.contains("kv 0 dump"), "KV-dump shown when verbose");
        assert!(out.contains("arch = bert"), "print_info shown when verbose");
        assert!(out.contains("offloaded 43/43 layers"));
        assert!(
            out.contains("Loaded model"),
            "engine DEBUG shown when verbose"
        );
    }
}
