use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

use super::*;

/// Serializes the tests that touch the PROCESS-WIDE engine-diagnostics buffer
/// (`record_engine_diagnostic` / `take_engine_diagnostics` / `EngineDiagnosticCapture`).
/// They share one global buffer, so running them concurrently lets one test's
/// captured lines leak into another's assertion. Each such test takes this lock for
/// its whole body. Poison-tolerant: a panic in one test must not wedge the rest.
static DIAG_TEST_LOCK: Mutex<()> = Mutex::new(());

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

/// The load-failure diagnostic capture: EVERY engine ERROR line for the load
/// window is retained in emission order (no heuristic pick), WARN/INFO and
/// non-engine events are excluded, and `take`/`clear` drain/reset the buffer.
/// One combined test so the process-wide buffer isn't raced by parallel tests
/// (this is the only test that installs the capturing layer).
#[test]
fn engine_diagnostics_capture_clear_and_drain() {
    let _diag_guard = DIAG_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_engine_diagnostics();
    let subscriber = tracing_subscriber::registry().with(EngineDiagnosticCapture);
    tracing::subscriber::with_default(subscriber, || {
        // The specific root cause — captured first.
        tracing::error!(target: ENGINE_TARGET, module = "llama.cpp::llama_model_load", "error loading model architecture: unknown model architecture: 'gemma4'");
        // The generic tail — captured too (kept for robustness, not discarded).
        tracing::error!(target: ENGINE_TARGET, module = "llama.cpp::llama_model_load_from_file_impl", "failed to load model");
        // Advisory WARN — must NOT pollute the failure reason.
        tracing::warn!(target: ENGINE_TARGET, "tokenizer config may be incorrect");
        // INFO — excluded.
        tracing::info!(target: ENGINE_TARGET, "offloaded 43/43 layers");
        // ERROR on a non-engine target — excluded.
        tracing::error!(target: "higgs", "host-side error");
    });
    assert_eq!(
        take_engine_diagnostics(),
        vec![
            "error loading model architecture: unknown model architecture: 'gemma4'".to_string(),
            "failed to load model".to_string(),
        ],
        "both engine ERROR lines in emission order; WARN/INFO/non-engine excluded"
    );
    // take() drained the buffer — a second take yields nothing.
    assert!(
        take_engine_diagnostics().is_empty(),
        "take drains the buffer"
    );
    // clear() resets even when new lines were captured since.
    tracing::subscriber::with_default(
        tracing_subscriber::registry().with(EngineDiagnosticCapture),
        || tracing::error!(target: ENGINE_TARGET, "stale"),
    );
    clear_engine_diagnostics();
    assert!(
        take_engine_diagnostics().is_empty(),
        "clear resets the buffer"
    );
}

/// `record_engine_diagnostic` honors the [`MAX_ENGINE_DIAGNOSTICS`] bound: lines
/// past the cap are dropped (the false branch of the `buf.len() < MAX` guard) so
/// a pathological engine can't grow the buffer without limit. The first lines
/// (the root cause) are the ones retained.
#[test]
fn record_engine_diagnostic_caps_at_max() {
    let _diag_guard = DIAG_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_engine_diagnostics();
    let total = MAX_ENGINE_DIAGNOSTICS + 10;
    for i in 0..total {
        record_engine_diagnostic(format!("line {i}"));
    }
    let captured = take_engine_diagnostics();
    assert_eq!(
        captured.len(),
        MAX_ENGINE_DIAGNOSTICS,
        "buffer is bounded by MAX_ENGINE_DIAGNOSTICS, excess dropped"
    );
    assert_eq!(
        captured.first().map(String::as_str),
        Some("line 0"),
        "the first (root-cause) lines are retained"
    );
    assert_eq!(
        captured.last().map(String::as_str),
        Some(format!("line {}", MAX_ENGINE_DIAGNOSTICS - 1).as_str()),
        "the last retained line is exactly at the cap boundary"
    );
    // No leakage into a subsequent window.
    clear_engine_diagnostics();
}

/// In normal mode an engine INFO event with NO `module` field passes the MODULE
/// gate (the `None => true` arm of `event_enabled`): the suppression list is
/// matched only on a present `module` value, so a module-less line is kept.
#[test]
fn normal_mode_keeps_engine_info_without_module_field() {
    let buf = BufWriter::default();
    let flag = Arc::new(AtomicBool::new(false));
    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(buf.clone())
            .with_filter(EngineLogFilter { verbose: flag }),
    );
    tracing::subscriber::with_default(subscriber, || {
        // Engine INFO with no `module` field at all -> visitor.module is None.
        tracing::info!(target: ENGINE_TARGET, "module-less engine line");
    });
    let out = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
    assert!(
        out.contains("module-less engine line"),
        "module-less engine INFO is kept (None arm of the MODULE gate)"
    );
}

/// In normal mode an engine INFO event whose `module` field arrives via `Debug`
/// (the `?value` sigil, not a string literal) is read by
/// [`ModuleVisitor::record_debug`]. A non-noisy debug module stays visible,
/// proving the Debug fallback populates `visitor.module` and the `Some` arm runs.
#[test]
fn normal_mode_reads_module_via_debug_fallback_keeps_non_noisy() {
    let buf = BufWriter::default();
    let flag = Arc::new(AtomicBool::new(false));
    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(buf.clone())
            .with_filter(EngineLogFilter { verbose: flag }),
    );
    let module_value = String::from("llama.cpp::load_tensors");
    tracing::subscriber::with_default(subscriber, || {
        // `?module_value` records the field via Debug, exercising record_debug.
        tracing::info!(target: ENGINE_TARGET, module = ?module_value, "debug-module engine line");
    });
    let out = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
    assert!(
        out.contains("debug-module engine line"),
        "non-noisy module read via Debug fallback stays visible"
    );
}

/// In normal mode an engine INFO event whose `module` field arrives via `Debug`
/// AND names a noisy module is still suppressed: the Debug fallback strips the
/// surrounding quotes so the value matches a [`NOISY_ENGINE_MODULES`] entry.
#[test]
fn normal_mode_reads_noisy_module_via_debug_fallback_suppresses() {
    let buf = BufWriter::default();
    let flag = Arc::new(AtomicBool::new(false));
    let subscriber = tracing_subscriber::registry().with(
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(buf.clone())
            .with_filter(EngineLogFilter { verbose: flag }),
    );
    let noisy = String::from("llama.cpp::print_info");
    tracing::subscriber::with_default(subscriber, || {
        tracing::info!(target: ENGINE_TARGET, module = ?noisy, "debug-noisy engine line");
    });
    let out = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
    assert!(
        !out.contains("debug-noisy engine line"),
        "noisy module read via Debug fallback is still suppressed (quotes stripped)"
    );
}

/// [`MessageVisitor::record_str`] is the defensive fallback for a `message` field
/// that arrives as a plain `&str` (rather than rendered format args via Debug).
/// An explicit string-valued `message` field on an engine ERROR is captured into
/// the diagnostics buffer through that `record_str` path.
#[test]
fn engine_diagnostics_capture_message_via_record_str() {
    let _diag_guard = DIAG_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_engine_diagnostics();
    let subscriber = tracing_subscriber::registry().with(EngineDiagnosticCapture);
    tracing::subscriber::with_default(subscriber, || {
        // An explicit `message = "..."` string field records via record_str.
        tracing::error!(target: ENGINE_TARGET, message = "explicit string message");
    });
    assert_eq!(
        take_engine_diagnostics(),
        vec!["explicit string message".to_string()],
        "engine ERROR with a string-valued message field is captured via record_str"
    );
    clear_engine_diagnostics();
}

/// [`set_engine_verbose`] is a no-op when logging was never installed (the
/// `ENGINE_VERBOSE` OnceLock is unset, so `get()` returns None). It must not
/// panic — the worker calls it from the `higgs/log_level` RPC unconditionally.
#[test]
fn set_engine_verbose_is_noop_when_uninstalled() {
    // ENGINE_VERBOSE is only set by install_worker_logging, which performs FFI
    // and installs a process-global subscriber; in the unit harness it is never
    // run, so this drives the None branch. Both values must be safe to call.
    set_engine_verbose(true);
    set_engine_verbose(false);
}
