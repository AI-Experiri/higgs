
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

/// The load-failure diagnostic capture: EVERY engine ERROR line for the load
/// window is retained in emission order (no heuristic pick), WARN/INFO and
/// non-engine events are excluded, and `take`/`clear` drain/reset the buffer.
/// One combined test so the process-wide buffer isn't raced by parallel tests
/// (this is the only test that installs the capturing layer).
#[test]
fn engine_diagnostics_capture_clear_and_drain() {
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
