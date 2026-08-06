//! `higgs model <search|show|download>` — the catalog from the command line.
//! Parsing and rendering are pure functions (unit-tested); the driver builds
//! its own runtime like the sibling fleet subcommands and talks to the Hub
//! through the same [`HfSource`] the facade ops use. Downloaded-state comes
//! from walking the default LM-Studio-layout scan roots (no worker, no
//! facade, so `show` prints sizes without fit verdicts — fit needs the
//! device sample only a running worker provides).

use std::io::{Error, Result, Write};

use crate::catalog::service::{self, LocalInventory};
use crate::catalog::source::HfSource;
use crate::catalog::wire::{CatalogModelDetail, CatalogQuery, CatalogSearchResponse, CatalogSort};
use crate::diagnostic::HiggsError;

const USAGE: &str = "usage: higgs model <search <query…> [--limit <n>] [--sort downloads|likes|updated|trending] | show <org/model> | download <org/model> [<file.gguf>]>";

/// One parsed `higgs model` invocation.
#[derive(Debug)]
pub(crate) enum ModelCmd {
    /// `search <query…>` — free-text catalog search.
    Search {
        query: String,
        /// `0` = the service default page size.
        limit: u32,
        sort: CatalogSort,
    },
    /// `show <org/model>` — one repo's detail (quants, sizes, README).
    Show { repo: String },
    /// `download <org/model> [<file>]` — pull a quant (default: the pick
    /// [`service::default_quant`] makes).
    Download { repo: String, file: Option<String> },
}

/// Parse `higgs model …` args. `Err` carries the full usage line plus what
/// was wrong — the caller prints it verbatim.
pub(crate) fn parse_model_cmd(args: &[String]) -> std::result::Result<ModelCmd, String> {
    let usage = |detail: &str| format!("{USAGE} ({detail})");
    match args.first().map(String::as_str) {
        Some("search") => {
            let mut words: Vec<String> = Vec::new();
            let mut limit = 0u32;
            let mut sort = CatalogSort::Downloads;
            let mut it = args[1..].iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--limit" => {
                        limit = it
                            .next()
                            .and_then(|v| v.parse().ok())
                            .ok_or_else(|| usage("--limit needs a number"))?;
                    }
                    "--sort" => {
                        sort = match it.next().map(String::as_str) {
                            Some("downloads") => CatalogSort::Downloads,
                            Some("likes") => CatalogSort::Likes,
                            Some("updated") => CatalogSort::Updated,
                            Some("trending") => CatalogSort::Trending,
                            _ => return Err(usage("--sort is downloads|likes|updated|trending")),
                        };
                    }
                    word => words.push(word.to_owned()),
                }
            }
            if words.is_empty() {
                return Err(usage("search needs a query"));
            }
            Ok(ModelCmd::Search {
                query: words.join(" "),
                limit,
                sort,
            })
        }
        Some("show") => args
            .get(1)
            .map(|repo| ModelCmd::Show { repo: repo.clone() })
            .ok_or_else(|| usage("show needs <org/model>")),
        Some("download") => {
            let repo = args
                .get(1)
                .cloned()
                .ok_or_else(|| usage("download needs <org/model>"))?;
            Ok(ModelCmd::Download {
                repo,
                file: args.get(2).cloned(),
            })
        }
        _ => Err(usage("unknown subcommand")),
    }
}

/// Human byte size (`512 B`, `4.0 KiB`, `4.3 GiB`).
pub(crate) fn fmt_bytes(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    #[allow(clippy::cast_precision_loss)]
    let b = bytes as f64;
    if bytes < 1024 {
        format!("{bytes} B")
    } else if b < KIB * KIB {
        format!("{:.1} KiB", b / KIB)
    } else if b < KIB * KIB * KIB {
        format!("{:.1} MiB", b / (KIB * KIB))
    } else {
        format!("{:.1} GiB", b / (KIB * KIB * KIB))
    }
}

/// The day part of a Hub ISO timestamp (`2026-07-01T10:00:00.000Z` → `2026-07-01`).
fn day(ts: &str) -> &str {
    ts.split('T').next().unwrap_or(ts)
}

/// Strip terminal-hostile bytes from Hub-derived text before it hits the
/// operator's terminal: control characters (ANSI escape introducers included)
/// are dropped, `\r` becomes `\n`, and only `\n`/`\t` survive as layout. A
/// README is arbitrary repo bytes — rendering it verbatim would hand the repo
/// author an escape-sequence channel into the terminal.
pub(crate) fn sanitize_terminal(s: &str) -> String {
    s.chars()
        .filter_map(|c| match c {
            '\n' | '\t' => Some(c),
            '\r' => Some('\n'),
            c if c.is_control() => None,
            c => Some(c),
        })
        .collect()
}

/// A `higgs model` failure as the io::Error the binary prints: the message may
/// embed Hub/mirror-derived detail, so it is control-stripped AND collapsed to
/// one line — a hostile origin can neither move the cursor nor fabricate extra
/// stderr lines.
fn sanitized_error(e: &HiggsError) -> Error {
    Error::other(table_cell(&sanitize_terminal(&e.to_string())))
}

/// One table cell from Hub-derived text: layout characters collapse to a
/// space so a hostile id/author/tag/file name can never fabricate table rows
/// or columns. (The README is the one place layout survives — it goes through
/// [`sanitize_terminal`] only.)
fn table_cell(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '\n' | '\t' | '\r' => ' ',
            c => c,
        })
        .collect()
}

/// Render search rows as an aligned table (a `downloaded` mark on rows any
/// quant of which is already local).
pub(crate) fn render_search(resp: &CatalogSearchResponse) -> String {
    if resp.models.is_empty() {
        return "no models found — try different words or a broader query\n".to_string();
    }
    let mut out = String::new();
    out.push_str(&format!(
        "{:<48} {:>12} {:>7} {:>11}\n",
        "MODEL", "DOWNLOADS", "LIKES", "UPDATED"
    ));
    for m in &resp.models {
        let downloads = m.downloads.map_or_else(|| "-".into(), |d| d.to_string());
        let likes = m.likes.map_or_else(|| "-".into(), |l| l.to_string());
        let updated = m.updated.as_deref().map_or("-", day);
        let mark = if m.downloaded { "  downloaded" } else { "" };
        out.push_str(&format!(
            "{:<48} {:>12} {:>7} {:>11}{}\n",
            table_cell(&m.id),
            downloads,
            likes,
            table_cell(updated),
            mark
        ));
    }
    out
}

/// Render one repo's detail: metadata line, the quant table (size, fit,
/// downloaded), more-by-author, then the README.
pub(crate) fn render_detail(d: &CatalogModelDetail) -> String {
    let mut out = String::new();
    out.push_str(&format!("{}\n", table_cell(&d.summary.id)));
    let mut meta: Vec<String> = Vec::new();
    if let Some(g) = &d.summary.gguf {
        if let Some(arch) = &g.arch {
            meta.push(format!("arch {}", table_cell(arch)));
        }
        if let Some(p) = g.params_total {
            meta.push(format!("params {}", fmt_count(p)));
        }
        if let Some(c) = g.ctx_train {
            meta.push(format!("ctx {c}"));
        }
    }
    if let Some(dl) = d.summary.downloads {
        meta.push(format!("downloads {dl}"));
    }
    if let Some(likes) = d.summary.likes {
        meta.push(format!("likes {likes}"));
    }
    if let Some(updated) = &d.summary.updated {
        meta.push(format!("updated {}", table_cell(day(updated))));
    }
    if !meta.is_empty() {
        out.push_str(&format!("  {}\n", meta.join(" · ")));
    }
    if !d.tags.is_empty() {
        let tags: Vec<String> = d.tags.iter().map(|t| table_cell(t)).collect();
        out.push_str(&format!("  tags: {}\n", tags.join(", ")));
    }
    out.push_str("\nfiles:\n");
    out.push_str(&format!(
        "  {:<44} {:>8} {:>9} {:>9}\n",
        "FILE", "QUANT", "SIZE", "FIT"
    ));
    for q in &d.quants {
        let size = q.size_bytes.map_or_else(|| "-".into(), fmt_bytes);
        let fit = match &q.fit {
            Some(f) if f.fits => "fits",
            Some(_) => "too large",
            None => "-",
        };
        let mark = if q.downloaded { "  downloaded" } else { "" };
        out.push_str(&format!(
            "  {:<44} {:>8} {:>9} {:>9}{}\n",
            table_cell(&q.file),
            table_cell(q.quant.as_deref().unwrap_or("-")),
            size,
            fit,
            mark
        ));
    }
    if !d.more_by_author.is_empty() {
        let author = table_cell(d.summary.author.as_deref().unwrap_or("this author"));
        out.push_str(&format!("\nmore by {author}:\n"));
        for m in &d.more_by_author {
            out.push_str(&format!("  {}\n", table_cell(&m.id)));
        }
    }
    if let Some(readme) = &d.readme {
        out.push_str("\nREADME:\n");
        out.push_str(readme);
        if !readme.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

/// Human count (`8030261248` → `8.0B`, `1234567` → `1.2M`).
fn fmt_count(n: u64) -> String {
    #[allow(clippy::cast_precision_loss)]
    let f = n as f64;
    if n >= 1_000_000_000 {
        format!("{:.1}B", f / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", f / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", f / 1e3)
    } else {
        n.to_string()
    }
}

/// `higgs model <…>`: parse, run on a fresh runtime, print. Errors return
/// non-`Ok` so the binary exits non-zero (it prints the message).
pub fn run_model(args: &[String]) -> Result<()> {
    let cmd = match parse_model_cmd(args) {
        Ok(cmd) => cmd,
        Err(usage) => {
            eprintln!("{usage}");
            return Err(Error::other("unknown model subcommand"));
        }
    };
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(run_cmd(cmd)).map_err(|e| sanitized_error(&e))
}

/// The local downloaded-state for CLI rows: walk every default
/// LM-Studio-layout scan root (which includes `~/.higgs/models`).
fn cli_inventory() -> LocalInventory {
    let mut inv = LocalInventory::empty();
    for dir in crate::HiggsConfig::default().lmstudio_dirs {
        inv.add_models_dir(&dir);
    }
    inv
}

async fn run_cmd(cmd: ModelCmd) -> std::result::Result<(), HiggsError> {
    match cmd {
        ModelCmd::Search { query, limit, sort } => {
            let q = CatalogQuery {
                search: query,
                author: None,
                sort: Some(sort),
                // `0` = "not given" from the parser → let the service default.
                limit: (limit > 0).then_some(limit),
                compatible_only: None,
            };
            // The CLI renders no fit column, so it passes no VRAM figure —
            // rows come back verdict-free rather than with an unused estimate.
            let resp = service::search(&HfSource, &q, &cli_inventory(), None).await?;
            print!("{}", sanitize_terminal(&render_search(&resp)));
            Ok(())
        }
        ModelCmd::Show { repo } => {
            let d = service::detail(&HfSource, &repo, &cli_inventory(), None).await?;
            print!("{}", sanitize_terminal(&render_detail(&d)));
            Ok(())
        }
        ModelCmd::Download { repo, file } => {
            let file = match file {
                Some(f) => f,
                None => {
                    // No file named: fetch the quant list and take the default
                    // pick (Q4_K_M → largest fitting → smallest).
                    let d = service::detail(&HfSource, &repo, &cli_inventory(), None).await?;
                    let pick = service::default_quant(&d.quants).ok_or_else(|| {
                        HiggsError::HubResourceNotFound {
                            repo: repo.clone(),
                            resource: "a .gguf file".to_string(),
                            detail: "the repo lists no .gguf files".to_string(),
                        }
                    })?;
                    let size = pick
                        .size_bytes
                        .map_or_else(|| "unknown size".into(), fmt_bytes);
                    // The pick is Hub-derived and not yet validated — sanitize
                    // before it touches the terminal.
                    eprintln!(
                        "no file named — picking {} ({size})",
                        sanitize_terminal(&pick.file)
                    );
                    pick.file.clone()
                }
            };
            let mut gate =
                crate::catalog::pull::ProgressGate::new(std::time::Duration::from_millis(100));
            let mut progress = |downloaded: u64, total: Option<u64>| {
                if !gate.should_emit(std::time::Instant::now()) {
                    return;
                }
                #[allow(clippy::cast_precision_loss)]
                let pct = total
                    .filter(|t| *t > 0)
                    .map(|t| format!(" ({:.0}%)", downloaded as f64 / t as f64 * 100.0))
                    .unwrap_or_default();
                let of = total
                    .map(|t| format!(" / {}", fmt_bytes(t)))
                    .unwrap_or_default();
                eprint!("\r  {}{of}{pct}   ", fmt_bytes(downloaded));
                let _ = std::io::stderr().flush();
            };
            let path = crate::catalog::pull::pull(&repo, &file, &mut progress).await?;
            eprintln!();
            println!("{}", path.display());
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "cli_tests.rs"]
mod tests;
