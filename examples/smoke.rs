//! Live smoke test for the in-process higgs crate API.
//!
//! Lists every scanned model, then loads + runs a real chat on the smallest
//! LLM(s) — proving control + chat work purely through the `Higgs` facade with
//! a REAL llama.cpp worker (no HTTP server). This example binary hosts the
//! `--higgs-worker` re-exec role itself, so `current_exe()` points back here and
//! real workers spawn.
//!
//! Run: `cargo run --example smoke`
//!   env SMOKE_N=<n>        how many smallest LLMs to test (default 1)
//!   env HIGGS_MODEL_DIR=…  extra LM-Studio-style scan root
//!   env RUST_LOG=higgs=info for load progress

use std::sync::Arc;
use std::time::Instant;

use higgs::{ChatDeltaKind, Higgs, HiggsConfig, SamplingParams};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

fn main() {
    // Worker role FIRST — before anything touches stdout (NDJSON JSON-RPC).
    if std::env::args().skip(1).any(|a| a == "--higgs-worker") {
        higgs::worker::worker_main();
        return;
    }

    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().with_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("higgs=info,warn")),
        ))
        .init();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("build tokio runtime");
    rt.block_on(run());
}

async fn run() {
    let mut config = HiggsConfig::default();
    if let Ok(dir) = std::env::var("HIGGS_MODEL_DIR") {
        if !dir.is_empty() {
            config.lmstudio_dirs.push(std::path::PathBuf::from(dir));
        }
    }
    let n: usize = std::env::var("SMOKE_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let higgs = Arc::new(Higgs::new(config));
    if let Err(e) = higgs.start().await {
        eprintln!("FATAL: higgs.start() failed: {e}");
        return;
    }

    // ── LIST ──────────────────────────────────────────────────────────────
    let entries = match higgs.model_entries().await {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FATAL: model_entries() failed: {e}");
            higgs.stop().await;
            return;
        }
    };
    println!(
        "\n══════════════════ SCANNED MODELS ({}) ══════════════════",
        entries.len()
    );
    let mut sorted: Vec<_> = entries.iter().collect();
    sorted.sort_by_key(|e| e.model.size_bytes);
    for e in &sorted {
        let gb = e.model.size_bytes as f64 / 1e9;
        println!(
            "  {:>7.2} GB  {}  {}  arch={} quant={} chat_template={}",
            gb,
            if e.model.has_chat_template {
                "LLM  "
            } else {
                "embed"
            },
            e.model.id,
            e.model.arch.clone().unwrap_or_else(|| "?".into()),
            e.model.quant.clone().unwrap_or_else(|| "?".into()),
            e.model.has_chat_template,
        );
    }

    // ── PICK LLMs (chat template = chat-capable; excludes embedders + mmproj) ─
    // Some embedding models (e.g. ollama qwen3-embedding) inherit a base chat
    // template in their GGUF, so has_chat_template alone isn't enough — exclude
    // anything named "embed".
    let llms: Vec<&higgs::HiggsModelEntry> = sorted
        .iter()
        .copied()
        .filter(|e| e.model.has_chat_template && !e.model.id.to_lowercase().contains("embed"))
        .collect();
    let targets: Vec<&higgs::HiggsModelEntry> = llms.into_iter().take(n).collect();
    if targets.is_empty() {
        println!("\nNo LLMs found to test.");
        higgs.stop().await;
        return;
    }

    println!(
        "\n══════════════════ TESTING {} LLM(s) ══════════════════",
        targets.len()
    );
    for e in targets {
        chat_one(&higgs, &e.model.id, e.model.size_bytes).await;
    }

    higgs.stop().await;
    println!("\nDone.");
}

async fn chat_one(higgs: &Arc<Higgs>, id: &str, size_bytes: u64) {
    println!(
        "\n──────── {}  ({:.2} GB) ────────",
        id,
        size_bytes as f64 / 1e9
    );

    let t_load = Instant::now();
    print!("  loading… ");
    if let Err(e) = higgs.load(id, None).await {
        println!("FAIL\n  ❌ load: {e}");
        return;
    }
    println!("ok in {:.1}s", t_load.elapsed().as_secs_f64());

    let messages = serde_json::json!([
        {"role": "user", "content": "Reply with a short one-sentence friendly greeting."}
    ])
    .to_string();

    // Generous budget so reasoning models (Nemotron-H, gemma-4) can think AND answer.
    let t_gen = Instant::now();
    let (mut rx, handle) = match higgs
        .chat_stream(
            id.to_string(),
            messages,
            512,
            SamplingParams::default(),
            None,
            None,
        )
        .await
    {
        Ok(pair) => pair,
        Err(e) => {
            println!("  ❌ chat_stream: {e}");
            let _ = higgs.unload().await;
            return;
        }
    };

    let mut content = String::new();
    let mut reasoning = String::new();
    while let Some(delta) = rx.recv().await {
        match delta.kind {
            ChatDeltaKind::Content => content.push_str(&delta.text),
            ChatDeltaKind::Reasoning => reasoning.push_str(&delta.text),
            ChatDeltaKind::ToolCall => {}
        }
    }

    match handle.await {
        Ok(Ok(outcome)) => {
            let secs = t_gen.elapsed().as_secs_f64();
            let tps = if secs > 0.0 {
                outcome.completion_tokens as f64 / secs
            } else {
                0.0
            };
            let answer = if outcome.content.trim().is_empty() {
                content.trim()
            } else {
                outcome.content.trim()
            };
            if !reasoning.trim().is_empty() {
                let think = reasoning.trim();
                let snip: String = think.chars().take(160).collect();
                println!(
                    "  💭 thought: {}{}",
                    snip,
                    if think.chars().count() > 160 {
                        "…"
                    } else {
                        ""
                    }
                );
            }
            println!("  💬 answer:  {answer}");
            let verdict = if answer.is_empty() {
                "⚠️  PASS (empty answer)"
            } else {
                "✅ PASS"
            };
            println!(
                "  {verdict}  {} tokens in {:.1}s ({:.1} tok/s), finish={}",
                outcome.completion_tokens, secs, tps, outcome.finish_reason
            );
        }
        Ok(Err(e)) => println!("  ❌ generation error: {e}"),
        Err(e) => println!("  ❌ chat task join error: {e}"),
    }

    let _ = higgs.unload().await;
}
