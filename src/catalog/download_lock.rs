//! Per-key MACHINE-WIDE download lock — the authority for "is anyone on this
//! machine downloading `(repo, file)` right now?"
//!
//! One lock file per key at `<models_root>/.download-locks/<safe_key>.lock`,
//! held via `fs2::try_lock_exclusive` (advisory `flock`) for the transfer's
//! entire lifetime. The KERNEL drops the lock on any process exit — Done,
//! Failed, Cancelled, SIGKILL, panic, power loss — so there is no residue
//! predicate to attack: the state is a kernel-managed file lock, not a
//! ledger row's on-disk story.
//!
//! Contrast with the LEDGER (`crate::catalog::ledger`), which stays pure
//! status/history — a row's `Downloading` value is a hint for the fleet
//! views, not a claim of ownership. All authority for "may I start this
//! transfer?" flows through THIS lock; the ledger just records what
//! happened.
//!
//! Per CLAUDE.md all file locking in this crate goes through `fs2::FileExt`.
//! No raw `libc::flock` in new code.
//!
//! # Safety key encoding
//!
//! `(repo, file)` is not a filesystem-safe name — `/` in the repo, arbitrary
//! length (dest_path allows a 255-byte org + 255-byte model + 218-byte file
//! → up to 737 bytes concatenated, way past NAME_MAX = 255) — and any
//! plain-text encoding using `--`/`__` separators is ambiguous within
//! dest_path's own `[A-Za-z0-9._-]` alphabet (`a/b--c` vs `a--b/c` both
//! flatten to `a--b--c`, so DIFFERENT valid keys would share a lock file
//! and one download would falsely refuse the other with HG090).
//!
//! Encoding: `sha256(repo || 0x00 || file)` → 64 hex chars; a
//! null-delimited byte sequence cannot collide across `(repo, file)` pairs
//! because null is banned in dest_path segments. The full 64-hex filename
//! is fixed-length (NAME_MAX-safe by construction) and collision-free
//! within cryptographic-hash strength.

use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use fs2::FileExt;
use sha2::{Digest, Sha256};

use crate::diagnostic::HiggsError;

/// The directory holding per-key `*.lock` files. Kept inside `models_root`
/// (same isolation the ledger enjoys: every test root is separate for free)
/// and hidden with a leading dot so the model scanner ignores it.
pub fn locks_dir(models_root: &Path) -> PathBuf {
    models_root.join(".download-locks")
}

/// Deterministic lock-file path for a `(repo, file)` key.
///
/// Uses `sha256(lower(repo) || 0x00 || lower(file))` → 64 hex chars. Null
/// byte is banned in every dest_path segment, so no two valid keys can
/// produce the same input byte sequence and the encoding is collision-free
/// within cryptographic-hash strength. Fixed length is NAME_MAX-safe
/// regardless of how long the raw key would be under dest_path's 255+255+218
/// per-segment budget.
///
/// CASE-FOLDED on purpose: the models dir on default macOS APFS is
/// case-INSENSITIVE, so `m.gguf` and `M.GGUF` are the SAME destination
/// file — two case-variant keys hashing to different locks would let two
/// writers race one rename target. ASCII-lowercasing before hashing makes
/// case variants share one lock, matching the filesystem's identity. On a
/// case-SENSITIVE filesystem (Linux ext4) this over-locks two genuinely
/// distinct files that differ only by case — safe (one refuses HG090) and
/// vanishingly rare (HF repos don't ship case-twin filenames). dest_path's
/// alphabet is `[A-Za-z0-9._-]`, so ASCII lowercasing is total.
pub fn lock_path(models_root: &Path, repo: &str, file: &str) -> PathBuf {
    let mut h = Sha256::new();
    h.update(repo.to_ascii_lowercase().as_bytes());
    h.update([0u8]);
    h.update(file.to_ascii_lowercase().as_bytes());
    let hex = h
        .finalize()
        .iter()
        .fold(String::with_capacity(64), |mut s, b| {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
            s
        });
    locks_dir(models_root).join(format!("{hex}.lock"))
}

/// RAII hold on a `(repo, file)` download slot.
///
/// While a `DownloadLock` is alive, no other process on this machine can
/// hold the same key (`fs2::try_lock_exclusive` returns `WouldBlock`). Drop
/// releases the flock; the kernel also releases it on any process exit.
#[must_use = "the lock is released when this guard is dropped"]
#[derive(Debug)]
pub struct DownloadLock {
    // Field must stay named (not `_`) so it lives for the guard's lifetime;
    // `unlock` on Drop uses it.
    file: File,
    path: PathBuf,
}

impl DownloadLock {
    /// Try to claim `(repo, file)` on `models_root`.
    ///
    /// Returns `Ok(guard)` when the lock is held by us; `Err(HG090)` when
    /// another live process on this machine already holds it; `Err(HG034)`
    /// for any other filesystem failure (lock dir uncreatable, lock file
    /// unopenable — a legitimate I/O error, NOT contention).
    pub fn acquire(models_root: &Path, repo: &str, file: &str) -> Result<Self, HiggsError> {
        let dir = locks_dir(models_root);
        std::fs::create_dir_all(&dir).map_err(|e| HiggsError::HubFileWrite {
            repo: repo.to_owned(),
            file: file.to_owned(),
            detail: format!("create download-locks dir {}: {e}", dir.display()),
        })?;
        let path = lock_path(models_root, repo, file);
        let f = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| HiggsError::HubFileWrite {
                repo: repo.to_owned(),
                file: file.to_owned(),
                detail: format!("open download lock {}: {e}", path.display()),
            })?;
        // Bounded retry to absorb the TRANSIENT window where the read-side
        // stale probe (`is_key_stale`) briefly holds the same flock — a
        // real acquirer arriving in that microsecond gap would otherwise
        // read as HG090 despite no legitimate holder existing. A genuine
        // in-flight transfer holds the lock for its entire lifetime
        // (seconds to minutes), so a real conflict persists past all
        // retries and still surfaces HG090. Total wait cap: ~50ms
        // (5×10ms), well under a hub RPC timeout, invisible in the UI.
        for _ in 0..5 {
            match f.try_lock_exclusive() {
                Ok(()) => return Ok(Self { file: f, path }),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                Err(e) => {
                    return Err(HiggsError::HubFileWrite {
                        repo: repo.to_owned(),
                        file: file.to_owned(),
                        detail: format!("flock download lock {}: {e}", path.display()),
                    });
                }
            }
        }
        Err(HiggsError::DownloadInFlight {
            repo: repo.to_owned(),
            file: file.to_owned(),
        })
    }

    /// The lock file's path (for observability / logging).
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// True iff this guard protects the `(models_root, repo, file)` key —
    /// i.e. its stored lock-file path equals what `lock_path` would compute
    /// for those inputs. Used by `download_dual_locked` to reject a caller
    /// that acquired a lock for one key and tried to use it for another
    /// (the guard alone does not carry the key text; it's just an fd holding
    /// a flock on a specific file, so identity check must be explicit).
    pub fn protects(&self, models_root: &Path, repo: &str, file: &str) -> bool {
        self.path == lock_path(models_root, repo, file)
    }
}

/// Does the lock FILE for `(repo, file)` exist at all? (No flock probe —
/// pure existence check; NOT proof of staleness on its own.) Used by the
/// ledger sweep's lockless-legacy-row aging: a `Downloading` row with no
/// lock file cannot belong to a current-build writer (they acquire the
/// lock BEFORE the ledger insert), so after a grace window it is reaped.
pub fn lock_file_exists(models_root: &Path, repo: &str, file: &str) -> bool {
    lock_path(models_root, repo, file).is_file()
}

/// Can we PROVE the machine-wide slot for `(repo, file)` is stale — i.e.
/// the lock file exists (someone downloaded this key at some point) AND
/// nobody currently holds the flock? A read-only, side-effect-free probe
/// (never creates the lock file, never touches the ledger, releases the
/// probe flock immediately).
///
/// Returns `true` ONLY when both conditions hold. Conservative on every
/// other case:
///
///  * lock file does not exist → `false` (either nobody ever downloaded
///    this key, or a test-scaffolded ledger row exists without a lock —
///    in either case we CANNOT prove staleness from a missing file, and
///    sweeping would clobber a legitimate row inserted before the lock
///    file was created).
///  * lock file exists and `try_lock_exclusive` returns `WouldBlock` →
///    `false` (live owner).
///  * any other filesystem error → `false` (can't prove).
///
/// The invariant matters: a read-side sweep MUST only fire when the
/// probe returns `true`, or a live download would be falsely
/// terminalized. Missing-file is deliberately NOT proof of staleness.
pub fn is_key_stale(models_root: &Path, repo: &str, file: &str) -> bool {
    let path = lock_path(models_root, repo, file);
    let f = match OpenOptions::new().read(true).write(true).open(&path) {
        Ok(f) => f,
        // File never existed → NOT proof of staleness (see doc).
        Err(_) => return false,
    };
    match f.try_lock_exclusive() {
        Ok(()) => {
            // Grabbed it → nobody was holding it → residue, safe to sweep.
            let _ = fs2::FileExt::unlock(&f);
            true
        }
        // Someone actively holds it OR probe failed → not proven stale.
        Err(_) => false,
    }
}

impl Drop for DownloadLock {
    fn drop(&mut self) {
        // Best-effort explicit unlock (kernel also releases on close).
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

#[cfg(test)]
#[path = "download_lock_tests.rs"]
mod tests;
