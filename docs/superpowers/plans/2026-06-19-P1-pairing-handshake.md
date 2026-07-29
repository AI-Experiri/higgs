# P1 — Pairing + Handshake (iroh Endpoint, auth, HELLO) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]`. After each task: run the task tests, then `scripts/coverage.sh`, then the **codex review loop until it converges** (`codex review --uncommitted`), then commit. Do not advance to the next task until codex review converges.

**Goal:** Two higgs binaries can pair over iroh and complete a versioned HELLO handshake: a node dials a hub using a one-time pairing token, the hub gates the connection post-HELLO against an `EndpointId` allowlist, agrees a protocol version, and persists the pairing. A stranger and a silent (handshake-stalled) peer are both rejected. **No chat, no workers, no inventory yet.**

**Architecture:** A persisted ed25519 `SecretKey` (`~/.higgs/endpoint.key`) gives each binary a stable iroh `EndpointId`. `src/auth.rs` owns the allowlist (`pairings.json`) and the pairing-token mint/burn — the crate's own serde, no external import (one-way dep rule). `src/node.rs` owns the iroh `Endpoint` bind, the accept loop, the post-HELLO gate (with a stalled-handshake deadline), and the node-side dialer. HELLO is the first `RpcFrame` on the control stream — `RpcRequest{method:"higgs/node/hello"}` answered by `RpcResponse` — reusing the existing `rpc.rs` NDJSON codec over iroh's raw bidi streams. Four new diagnostics (HG022/023/024/028) name the four gate outcomes.

**Tech Stack:** Rust, `iroh` 1.0 / `iroh-base` 1.0 / `iroh-tickets` 1.0 (QUIC p2p), existing `dirs`/`rand`/`serde`/`tokio`, existing `rpc::RpcFrame`. Builds on P0's `src/actor.rs`.

**iroh 1.0 API (per DESIGN-remote.md §3, verified 2026-06-19 — Task 1 re-confirms against the compiler):**
- Identity = `EndpointId` (= `PublicKey`); **iroh has no `NodeId`**. Our `NodeId` newtype wraps it (P4).
- `Endpoint::builder(presets::N0).secret_key(sk).alpns(vec![ALPN.to_vec()]).bind().await` → `Endpoint`; `endpoint.id() -> EndpointId`.
- `SecretKey::from_bytes(&[u8;32])` (infallible, by ref) ↔ `to_bytes() -> [u8;32]`; `SecretKey::generate(&mut rand::rngs::OsRng)`.
- `endpoint.connect(target, ALPN).await` where target is `EndpointId` (reconnect) or `EndpointAddr` (first contact via ticket).
- `endpoint.accept().await -> Option<Incoming>` → `incoming.await? -> Connection`.
- `conn.remote_id() -> EndpointId` (cryptographic peer identity).
- `conn.open_bi()/accept_bi().await -> Result<(SendStream, RecvStream), ConnectionError>` (named futures). `SendStream: AsyncWrite`, `RecvStream: AsyncRead` — boxable.
- `iroh_tickets::endpoint::EndpointTicket` — `.endpoint_addr() -> &EndpointAddr`; `Display`/`FromStr` for string round-trip. `EndpointAddr::new(id).with_relay_url(url)`.
- `accept_bi()` only fires after the opener writes — the dialer (node) sends HELLO first.

**Contract:** existing tests stay green; coverage stays ≥ 90%. New network code is covered by an integration test (`tests/remote_pairing.rs`) that runs two in-process endpoints (relay disabled, local discovery) so it needs no external relay.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `Cargo.toml` | Add `iroh`, `iroh-base`, `iroh-tickets` (all `1.0`). | modify |
| `src/home.rs` | NEW. `higgs_home() -> PathBuf` (`~/.higgs`, `HIGGS_HOME` override), `ensure_home()`. One home, one place. | create |
| `src/node/identity.rs` | NEW. `load_or_create_secret(path) -> SecretKey`; `bind_endpoint(sk) -> Endpoint`. The iroh identity + bind. | create |
| `src/auth.rs` | NEW. `Allowlist` (pairings.json: load/save/add/contains/remove) + `PairingTokens` (mint/validate/burn; TTL + single-use). Crate's own serde. | create |
| `src/remote.rs` | NEW. Protocol constants (`ALPN`, `M_HELLO`), `HelloParams`/`HelloResult` serde types, `negotiate_version`. The wire vocabulary. | create |
| `src/node/mod.rs` | NEW. `Endpoint` accept loop + post-HELLO gate + stalled-handshake deadline (hub side); `dial_and_hello` (node side). Uses `rpc.rs` over bidi streams. | create |
| `src/diagnostic.rs` | Append HG022/HG023/HG024/HG028 after HG021. | modify |
| `src/lib.rs` | Declare `mod home; mod auth; mod remote; mod node;`. | modify |
| `src/bin/higgs.rs` | Add `--node` role arm + `link`/`node connect` subcommands (minimal). | modify |
| `tests/remote_pairing.rs` | NEW. Integration: pair OK, stranger rejected (HG024), stalled peer dropped (HG028), version mismatch (HG023). | create |

> **Module layout note:** `src/node/` is a directory module (`mod.rs` + `identity.rs`); `auth.rs`/`remote.rs`/`home.rs` are flat. This keeps the iroh-heavy node code in one folder while auth/wire stay independently testable.

---

## Task 1: Deps + home + iroh identity + Endpoint bind (the API gateway)

This task de-risks the whole phase: add the crates and confirm the iroh 1.0 API names against the real compiler, ending with two endpoints that bind with stable ids.

**Files:**
- Modify: `Cargo.toml`, `src/lib.rs`
- Create: `src/home.rs`, `src/node/identity.rs`, `src/node/mod.rs` (skeleton)
- Test: inline in `home.rs` + `node/identity.rs`

- [ ] **Step 1: Add the iroh dependencies**

Add to `Cargo.toml [dependencies]` (alphabetical, near existing entries):

```toml
iroh = "1.0"
iroh-base = "1.0"
iroh-tickets = "1.0"
```

Run: `cargo fetch 2>&1 | tail -5` — Expected: resolves cleanly. If MSRV errors appear, note the required rustc and confirm the toolchain (`rustc --version`).

- [ ] **Step 2: Declare the new modules**

In `src/lib.rs`, after `pub mod actor;`, add:

```rust
pub mod auth;
pub mod home;
pub mod node;
pub mod remote;
```

- [ ] **Step 3: Write `src/home.rs` with a failing test**

```rust
//! Single home for all higgs identity/state: `~/.higgs` (override with `HIGGS_HOME`).

use std::path::PathBuf;

/// The higgs home directory: `$HIGGS_HOME` if set, else `~/.higgs`.
pub fn higgs_home() -> PathBuf {
    if let Some(over) = std::env::var_os("HIGGS_HOME") {
        return PathBuf::from(over);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".higgs")
}

/// Create the home directory if absent; returns its path.
pub fn ensure_home() -> std::io::Result<PathBuf> {
    let home = higgs_home();
    std::fs::create_dir_all(&home)?;
    Ok(home)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honors_higgs_home_override() {
        // SAFETY: single-threaded test; restore after.
        let tmp = std::env::temp_dir().join("higgs-home-test-xyz");
        unsafe { std::env::set_var("HIGGS_HOME", &tmp) };
        assert_eq!(higgs_home(), tmp);
        unsafe { std::env::remove_var("HIGGS_HOME") };
    }
}
```

Run: `cargo test --lib home:: 2>&1 | tail -5` — Expected: PASS.

- [ ] **Step 4: Write `src/node/identity.rs` — secret persistence + bind, with a failing test**

```rust
//! iroh identity: a persisted ed25519 SecretKey → a stable EndpointId across restarts.

use std::path::Path;

use iroh::{Endpoint, SecretKey};

use crate::remote::ALPN;

/// Load the 32 secret bytes from `path`, or generate + persist them (chmod 0600).
/// The `EndpointId` derived from this key is stable across restarts.
pub fn load_or_create_secret(path: &Path) -> std::io::Result<SecretKey> {
    if let Ok(bytes) = std::fs::read(path) {
        if let Ok(arr) = <[u8; 32]>::try_from(bytes.as_slice()) {
            return Ok(SecretKey::from_bytes(&arr));
        }
        // Corrupt key file: fail loudly rather than silently regenerating a new id.
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "endpoint.key is not 32 bytes",
        ));
    }
    let sk = SecretKey::generate(&mut rand::rngs::OsRng);
    std::fs::write(path, sk.to_bytes())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    }
    Ok(sk)
}

/// Bind an iroh Endpoint with a stable id and our ALPN, ready to dial and accept.
pub async fn bind_endpoint(sk: SecretKey) -> anyhow::Result<Endpoint> {
    let endpoint = Endpoint::builder(iroh::endpoint::presets::N0)
        .secret_key(sk)
        .alpns(vec![ALPN.to_vec()])
        .bind()
        .await?;
    Ok(endpoint)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_roundtrips_to_stable_id() {
        let dir = std::env::temp_dir().join("higgs-key-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("endpoint.key");
        let _ = std::fs::remove_file(&path);

        let sk1 = load_or_create_secret(&path).unwrap();
        let sk2 = load_or_create_secret(&path).unwrap(); // reads the same file
        assert_eq!(sk1.public(), sk2.public(), "id stable across loads");
        let _ = std::fs::remove_file(&path);
    }
}
```

> `anyhow` is already a dep (the project uses it). If not, use `Result<Endpoint, iroh::endpoint::BindError>`. `sk.public()` returns the `EndpointId`/`PublicKey` — confirm the accessor name against the compiler (`endpoint.id()` for the bound endpoint; `sk.public()` for the key). Adjust if the 1.0 API differs; this task's purpose is to lock the real names.

- [ ] **Step 5: `src/node/mod.rs` skeleton (declares the submodule, holds nothing yet)**

```rust
//! Node + hub iroh transport: bind, accept-loop gate, dial. Built out across P1–P3.

pub mod identity;
```

- [ ] **Step 6: Build + test + verify the API compiled**

Run: `cargo build 2>&1 | tail -20` — Expected: clean. Every iroh API name in Steps 4–5 must compile; if any differ from the design's reference, fix here and note the correction in a comment.
Run: `cargo test --lib home:: node::identity:: 2>&1 | tail -8` — Expected: PASS.

> `remote::ALPN` is referenced but not defined until Task 4. To keep Task 1 compiling, temporarily define `pub const ALPN: &[u8] = b"higgs/remote/1";` in a minimal `src/remote.rs` now (Task 4 fills the rest). Create `src/remote.rs` with just that const + module doc for this task.

- [ ] **Step 7: Codex review loop, then commit**

`codex review --uncommitted` → converge. Then:

```bash
git add Cargo.toml Cargo.lock src/lib.rs src/home.rs src/node/ src/remote.rs
git commit -m "feat(node): iroh deps + ~/.higgs home + persisted SecretKey + Endpoint bind"
```

---

## Task 2: `src/auth.rs` — the allowlist (`pairings.json`)

Pure logic, fully TDD-able, no network. The allowlist is the set of paired `EndpointId`s the hub admits.

**Files:**
- Create: `src/auth.rs` (Allowlist half)
- Test: inline

- [ ] **Step 1: Write failing tests**

```rust
//! Surface A auth — the machine allowlist (pairings.json) + pairing tokens.
//! Owned INSIDE the crate (crate's own serde); no common/engine/jigglebot import.

#[cfg(test)]
mod tests {
    use super::*;

    fn id_str() -> String {
        // 64 hex chars is a plausible EndpointId string; Allowlist stores ids as
        // their canonical String form (z-base-32 / hex per iroh Display).
        "aa".repeat(32)
    }

    #[test]
    fn add_contains_remove_roundtrip() {
        let dir = std::env::temp_dir().join("higgs-allow-test-1");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pairings.json");
        let _ = std::fs::remove_file(&path);

        let mut allow = Allowlist::load(&path).unwrap();
        assert!(!allow.contains(&id_str()));
        allow.add(id_str(), Some("studio-mac".into())).unwrap();
        assert!(allow.contains(&id_str()));

        // persisted: a fresh load sees it
        let reloaded = Allowlist::load(&path).unwrap();
        assert!(reloaded.contains(&id_str()));

        allow.remove(&id_str()).unwrap();
        assert!(!allow.contains(&id_str()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn load_missing_file_is_empty() {
        let path = std::env::temp_dir().join("higgs-allow-absent.json");
        let _ = std::fs::remove_file(&path);
        let allow = Allowlist::load(&path).unwrap();
        assert!(!allow.contains(&id_str()));
    }
}
```

Run: `cargo test --lib auth::tests::add_contains 2>&1 | tail -5` — Expected: FAIL (no `Allowlist`).

- [ ] **Step 2: Implement the `Allowlist`**

Prepend to `src/auth.rs` (above the test module):

```rust
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// One paired node: its EndpointId (canonical string) → an optional human label.
#[derive(Debug, Default, Serialize, Deserialize)]
struct PairingsFile {
    /// EndpointId string → label.
    nodes: BTreeMap<String, Option<String>>,
}

/// The Surface A allowlist: paired `EndpointId`s the hub admits. Backed by
/// `pairings.json`; mutations persist immediately (small file, infrequent writes).
pub struct Allowlist {
    path: PathBuf,
    file: PairingsFile,
}

impl Allowlist {
    /// Load from `path`; a missing file is an empty allowlist.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes).map_err(|e| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, e)
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => PairingsFile::default(),
            Err(e) => return Err(e),
        };
        Ok(Self { path: path.to_path_buf(), file })
    }

    /// Is this EndpointId (canonical string) paired?
    pub fn contains(&self, id: &str) -> bool {
        self.file.nodes.contains_key(id)
    }

    /// Add a paired id (idempotent); persists.
    pub fn add(&mut self, id: String, label: Option<String>) -> std::io::Result<()> {
        self.file.nodes.insert(id, label);
        self.save()
    }

    /// Remove a paired id (revocation); persists.
    pub fn remove(&mut self, id: &str) -> std::io::Result<()> {
        self.file.nodes.remove(id);
        self.save()
    }

    fn save(&self) -> std::io::Result<()> {
        let bytes = serde_json::to_vec_pretty(&self.file)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&self.path, bytes)
    }
}
```

Run: `cargo test --lib auth::tests 2>&1 | tail -5` — Expected: PASS.

- [ ] **Step 3: Codex review loop, then commit**

`codex review --uncommitted` → converge. Then:

```bash
git add src/auth.rs
git commit -m "feat(auth): Surface A allowlist — pairings.json load/add/contains/remove"
```

---

## Task 3: `src/auth.rs` — pairing tokens (mint / validate / burn)

A pairing token (`htk_<random>`) is single-use + TTL-bounded; it admits a not-yet-allowlisted id exactly once. Pure logic + a clock injected for deterministic TTL tests.

**Files:**
- Modify: `src/auth.rs`
- Test: inline

- [ ] **Step 1: Write failing tests**

```rust
    #[test]
    fn token_mint_validate_burn() {
        let mut tokens = PairingTokens::new();
        let tok = tokens.mint(now_ms(), 600_000); // 10 min TTL
        assert!(tok.starts_with("htk_"));
        // valid before burn
        assert!(tokens.validate_and_burn(&tok, now_ms()).is_ok());
        // single-use: second validation fails
        assert!(matches!(
            tokens.validate_and_burn(&tok, now_ms()),
            Err(TokenError::UnknownOrUsed)
        ));
    }

    #[test]
    fn token_expires() {
        let mut tokens = PairingTokens::new();
        let minted_at = 1_000_000u64;
        let tok = tokens.mint(minted_at, 600_000);
        // 11 minutes later
        assert!(matches!(
            tokens.validate_and_burn(&tok, minted_at + 660_000),
            Err(TokenError::Expired)
        ));
    }

    fn now_ms() -> u64 {
        1_750_000_000_000
    }
```

Run: `cargo test --lib auth::tests::token 2>&1 | tail -6` — Expected: FAIL.

- [ ] **Step 2: Implement `PairingTokens`**

Add to `src/auth.rs`:

```rust
use std::collections::HashMap;

/// Why a pairing token was rejected (maps to HG022 at the gate).
#[derive(Debug, PartialEq, Eq)]
pub enum TokenError {
    /// Token never minted, or already burned (single-use).
    UnknownOrUsed,
    /// Token minted but its TTL has elapsed.
    Expired,
}

/// In-memory mint/burn store for one-time pairing tokens. Not persisted: tokens
/// are short-lived (default 10 min) and a hub restart simply invalidates pending
/// ones — the operator re-mints. Single home for token state on the hub.
#[derive(Default)]
pub struct PairingTokens {
    /// token string → expiry epoch-ms.
    live: HashMap<String, u64>,
}

impl PairingTokens {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint a single-use token valid for `ttl_ms` from `now_ms`. Returns `htk_<hex>`.
    pub fn mint(&mut self, now_ms: u64, ttl_ms: u64) -> String {
        // 16 random bytes → hex. `rand` is already a dep.
        use rand::RngCore;
        let mut raw = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut raw);
        let tok = format!("htk_{}", hex_encode(&raw));
        self.live.insert(tok.clone(), now_ms.saturating_add(ttl_ms));
        tok
    }

    /// Validate a presented token and burn it (single-use). `Ok(())` admits the peer.
    pub fn validate_and_burn(&mut self, token: &str, now_ms: u64) -> Result<(), TokenError> {
        match self.live.remove(token) {
            None => Err(TokenError::UnknownOrUsed),
            Some(expiry) if now_ms > expiry => Err(TokenError::Expired),
            Some(_) => Ok(()),
        }
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}
```

Run: `cargo test --lib auth::tests 2>&1 | tail -6` — Expected: PASS (all auth tests).

- [ ] **Step 3: Codex review loop, then commit**

`codex review --uncommitted` → converge. Then:

```bash
git add src/auth.rs
git commit -m "feat(auth): one-time pairing tokens — mint/validate/burn with TTL"
```

---

## Task 4: `src/remote.rs` — HELLO types + version negotiation

The wire vocabulary: ALPN, the HELLO method const, the HELLO param/result serde shapes, and the pure version-negotiation function.

**Files:**
- Modify: `src/remote.rs` (created minimally in Task 1)
- Test: inline

- [ ] **Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negotiate_picks_max_common_version() {
        // node speaks [1], hub speaks [1] → agree 1
        assert_eq!(negotiate_version(&[1], 1, &[1], 1), Ok(1));
        // node speaks [1,2], hub speaks [1] → agree 1
        assert_eq!(negotiate_version(&[1, 2], 1, &[1], 1), Ok(1));
    }

    #[test]
    fn negotiate_fails_with_no_overlap() {
        // node speaks [2], min 2; hub speaks [1], min 1 → no agreed ≥ both mins
        assert_eq!(
            negotiate_version(&[2], 2, &[1], 1),
            Err(VersionMismatch { peer: vec![2], ours: vec![1] })
        );
    }

    #[test]
    fn hello_params_roundtrip_json() {
        let p = HelloParams {
            role: "node".into(),
            node_id: "z32id".into(),
            pairing_token: Some("htk_abc".into()),
            protocol_versions: vec![1],
            min_supported: 1,
            software_version: "0.4.2".into(),
        };
        let s = serde_json::to_string(&p).unwrap();
        let back: HelloParams = serde_json::from_str(&s).unwrap();
        assert_eq!(back.node_id, "z32id");
        assert_eq!(back.pairing_token.as_deref(), Some("htk_abc"));
    }
}
```

Run: `cargo test --lib remote::tests 2>&1 | tail -6` — Expected: FAIL.

- [ ] **Step 2: Implement `src/remote.rs`** (replace the Task-1 stub body, keep `ALPN`)

```rust
//! The remote wire vocabulary: ALPN, the `higgs/node/*` HELLO method, its serde
//! payloads, and version negotiation. Additive over the existing `rpc.rs` wire.

use serde::{Deserialize, Serialize};

/// QUIC ALPN for the higgs remote protocol.
pub const ALPN: &[u8] = b"higgs/remote/1";

/// HELLO — first control-stream frame (node → hub). See DESIGN-remote.md §4.1.
pub const M_HELLO: &str = "higgs/node/hello";

/// The wire-protocol majors this build speaks.
pub const PROTOCOL_VERSIONS: &[u32] = &[1];
/// The lowest major this build still accepts.
pub const MIN_SUPPORTED: u32 = 1;

/// node → hub HELLO request params.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloParams {
    pub role: String, // "node" | "hub"
    pub node_id: String, // self EndpointId (canonical string); MUST equal the QUIC peer id
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing_token: Option<String>, // only on first join
    pub protocol_versions: Vec<u32>,
    pub min_supported: u32,
    pub software_version: String,
}

/// hub → node HELLO result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelloResult {
    pub role: String,
    pub node_id: String,
    pub agreed_version: u32,
    pub software_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub assigned_label: Option<String>,
}

/// No agreed protocol version (maps to HG023, a fatal typed close).
#[derive(Debug, PartialEq, Eq)]
pub struct VersionMismatch {
    pub peer: Vec<u32>,
    pub ours: Vec<u32>,
}

/// Agree the single major both sides pin: max of the intersection, provided it is
/// ≥ both sides' `min_supported`. Open `capabilities` maps never gate (forward-compat).
pub fn negotiate_version(
    peer_versions: &[u32],
    peer_min: u32,
    our_versions: &[u32],
    our_min: u32,
) -> Result<u32, VersionMismatch> {
    let agreed = peer_versions
        .iter()
        .filter(|v| our_versions.contains(v))
        .copied()
        .max();
    match agreed {
        Some(v) if v >= peer_min && v >= our_min => Ok(v),
        _ => Err(VersionMismatch {
            peer: peer_versions.to_vec(),
            ours: our_versions.to_vec(),
        }),
    }
}
```

Run: `cargo test --lib remote::tests 2>&1 | tail -6` — Expected: PASS.

- [ ] **Step 3: Codex review loop, then commit**

`codex review --uncommitted` → converge. Then:

```bash
git add src/remote.rs
git commit -m "feat(remote): HELLO params/result + version negotiation"
```

---

## Task 5: Diagnostics HG022/HG023/HG024/HG028

Append four codes after `HG021` (`diagnostic.rs`), following the existing snafu/miette style. These name the four post-HELLO gate outcomes (DESIGN-remote.md §7.1).

**Files:**
- Modify: `src/diagnostic.rs`
- Test: inline (mirror any existing per-code display test)

- [ ] **Step 1: Read the HG021 variant to copy its exact style**

Run: `rg -n "HG021|HG020" src/diagnostic.rs` and read the surrounding `#[snafu(...)] / #[diagnostic(code(...))]` variant block so the new ones match field/attribute style exactly.

- [ ] **Step 2: Add the four variants**

In the `HiggsError` enum (after the HG021 variant), add (adapt field syntax to the enum's actual style — `#[snafu(display(...))]` + `#[diagnostic(code(...))]`, `severity(Error)` only on the fatal one):

```rust
    /// HG022 — a presented pairing token was expired, used, or unknown.
    #[snafu(display("[HG022] pairing token invalid (expired, used, or unknown): {detail}"))]
    #[diagnostic(code(HG022))]
    PairingTokenInvalid { detail: String },

    /// HG023 — no agreed protocol version (fatal, typed close).
    #[snafu(display("[HG023] no agreed protocol version: peer speaks {peer:?}, we accept {ours:?}"))]
    #[diagnostic(code(HG023), severity(Error))]
    VersionMismatch { peer: Vec<u32>, ours: Vec<u32> },

    /// HG024 — peer not in the allowlist and presented no valid pairing token.
    #[snafu(display("[HG024] peer {endpoint_id} is not in the allowlist and presented no valid pairing token"))]
    #[diagnostic(code(HG024))]
    NotAllowlisted { endpoint_id: String },

    /// HG028 — QUIC completed but no HELLO arrived within the deadline.
    #[snafu(display("[HG028] peer {endpoint_id} completed QUIC but sent no HELLO within {window}s; dropped"))]
    #[diagnostic(code(HG028))]
    HandshakeStalled { endpoint_id: String, window: u64 },
```

> Check whether `HiggsError` variants carry a `snafu(source)` / backtrace convention; these four are constructed directly (no source), like HG021's family. Match the enum's existing derive set.

- [ ] **Step 3: Build + a display test**

Add an inline test (matching any existing diagnostic test style):

```rust
    #[test]
    fn new_remote_codes_render() {
        let e = HiggsError::HandshakeStalled { endpoint_id: "z32".into(), window: 5 };
        assert!(e.to_string().starts_with("[HG028]"));
        let v = HiggsError::VersionMismatch { peer: vec![2], ours: vec![1] };
        assert!(v.to_string().starts_with("[HG023]"));
    }
```

Run: `cargo build 2>&1 | tail -10 && cargo test --lib diagnostic 2>&1 | tail -6` — Expected: PASS.

- [ ] **Step 4: Codex review loop, then commit**

`codex review --uncommitted` → converge. Then:

```bash
git add src/diagnostic.rs
git commit -m "feat(diag): HG022-HG024 + HG028 — post-HELLO gate outcomes"
```

---

## Task 6: `src/node/mod.rs` — accept-loop gate + dial (the handshake)

Wire it together: the hub binds + accepts, reads HELLO over a bidi stream (reusing `rpc.rs`), gates post-HELLO with a stalled-handshake deadline, negotiates version, persists the pairing, replies. The node dials + sends HELLO first. Covered by an integration test with relay disabled.

**Files:**
- Modify: `src/node/mod.rs`
- Create: `tests/remote_pairing.rs`

- [ ] **Step 1: Implement the gate + dial**

Add to `src/node/mod.rs`. Constants `HELLO_DEADLINE = Duration::from_secs(5)`. The control-stream frame I/O reuses `rpc::encode`/`rpc::decode` over `tokio::io::BufReader` on the `RecvStream` and writes to the `SendStream`. Key functions:

```rust
use std::time::Duration;

use iroh::endpoint::Connection;
use iroh::Endpoint;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::auth::{Allowlist, PairingTokens, TokenError};
use crate::diagnostic::HiggsError;
use crate::remote::{
    negotiate_version, HelloParams, HelloResult, M_HELLO, MIN_SUPPORTED, PROTOCOL_VERSIONS,
};
use crate::rpc::{self, RpcFrame, RpcRequest, RpcResponse};

/// Max time from accept() to a complete HELLO before we drop the conn (HG028).
pub const HELLO_DEADLINE: Duration = Duration::from_secs(5);

/// Outcome of gating one inbound connection.
#[derive(Debug, PartialEq, Eq)]
pub enum GateOutcome {
    /// Admitted: already-allowlisted peer, or a valid pairing token (now burned + added).
    Admitted { agreed_version: u32 },
    /// Rejected with a typed diagnostic (HG022/HG023/HG024/HG028).
    Rejected,
}

/// Read the first frame off the control stream as a HELLO, bounded by HELLO_DEADLINE.
/// Returns the parsed params, or `None` if it stalled / was malformed (caller → HG028).
async fn read_hello(recv: &mut iroh::endpoint::RecvStream) -> Option<(u64, HelloParams)> {
    let mut lines = BufReader::new(recv).lines();
    let first = tokio::time::timeout(HELLO_DEADLINE, lines.next_line()).await;
    let line = match first {
        Ok(Ok(Some(line))) => line,
        _ => return None, // timeout, EOF, or io error → stalled
    };
    match rpc::decode(&line) {
        Ok(RpcFrame::Request(req)) if req.method == M_HELLO => {
            serde_json::from_value::<HelloParams>(req.params).ok().map(|p| (req.id, p))
        }
        _ => None,
    }
}

/// Hub side: gate one accepted connection. `now_ms` is injected for testable TTLs.
pub async fn gate_connection(
    conn: &Connection,
    allow: &mut Allowlist,
    tokens: &mut PairingTokens,
    now_ms: u64,
    label_for_new: Option<String>,
) -> GateOutcome {
    let peer = conn.remote_id().to_string();
    let (send, mut recv) = match conn.accept_bi().await {
        Ok(pair) => pair,
        Err(_) => return GateOutcome::Rejected,
    };

    let Some((id, hello)) = read_hello(&mut recv).await else {
        // HG028 — QUIC done, no HELLO in time.
        let _ = HiggsError::HandshakeStalled { endpoint_id: peer.clone(), window: HELLO_DEADLINE.as_secs() };
        conn.close(0u32.into(), b"HG028");
        return GateOutcome::Rejected;
    };

    // 1. version negotiation (HG023, fatal)
    let agreed = match negotiate_version(
        &hello.protocol_versions, hello.min_supported, PROTOCOL_VERSIONS, MIN_SUPPORTED,
    ) {
        Ok(v) => v,
        Err(_) => {
            conn.close(0u32.into(), b"HG023");
            return GateOutcome::Rejected;
        }
    };

    // 2. allowlist OR valid pairing token
    let admitted = if allow.contains(&peer) {
        true
    } else if let Some(tok) = &hello.pairing_token {
        match tokens.validate_and_burn(tok, now_ms) {
            Ok(()) => {
                let _ = allow.add(peer.clone(), label_for_new.clone());
                true
            }
            Err(TokenError::Expired) | Err(TokenError::UnknownOrUsed) => false, // HG022
        }
    } else {
        false // HG024
    };

    if !admitted {
        conn.close(0u32.into(), b"HG024");
        return GateOutcome::Rejected;
    }

    // 3. reply HelloResult
    let result = HelloResult {
        role: "hub".into(),
        node_id: conn_local_id_string(conn),
        agreed_version: agreed,
        software_version: env!("CARGO_PKG_VERSION").into(),
        assigned_label: label_for_new,
    };
    let resp = RpcResponse {
        jsonrpc: "2.0".into(),
        id,
        result: Some(serde_json::to_value(result).unwrap()),
        error: None,
    };
    let mut send = send;
    let line = rpc::encode(&RpcFrame::Response(resp));
    let _ = send.write_all(format!("{line}\n").as_bytes()).await;
    let _ = send.flush().await;
    GateOutcome::Admitted { agreed_version: agreed }
}

fn conn_local_id_string(_conn: &Connection) -> String {
    // The hub's own EndpointId is known from its Endpoint; the integration test
    // passes it in via HelloResult assembly. Kept as a helper seam for P2.
    String::new()
}

/// Node side: dial `target`, open a control bi-stream, send HELLO first, await result.
pub async fn dial_and_hello(
    endpoint: &Endpoint,
    target: iroh::EndpointAddr,
    self_id: String,
    pairing_token: Option<String>,
) -> anyhow::Result<HelloResult> {
    let conn = endpoint.connect(target, crate::remote::ALPN).await?;
    let (mut send, recv) = conn.open_bi().await?;
    let params = HelloParams {
        role: "node".into(),
        node_id: self_id,
        pairing_token,
        protocol_versions: PROTOCOL_VERSIONS.to_vec(),
        min_supported: MIN_SUPPORTED,
        software_version: env!("CARGO_PKG_VERSION").into(),
    };
    let req = RpcRequest { jsonrpc: "2.0".into(), id: 1, method: M_HELLO.into(), params: serde_json::to_value(params)? };
    send.write_all(format!("{}\n", rpc::encode(&RpcFrame::Request(req))).as_bytes()).await?;
    send.flush().await?;

    let mut lines = BufReader::new(recv).lines();
    let line = lines.next_line().await?.ok_or_else(|| anyhow::anyhow!("no HELLO reply"))?;
    match rpc::decode(&line)? {
        RpcFrame::Response(resp) => {
            if let Some(err) = resp.error {
                anyhow::bail!("hub rejected HELLO: {}", err.message);
            }
            Ok(serde_json::from_value(resp.result.unwrap_or_default())?)
        }
        _ => anyhow::bail!("unexpected reply frame"),
    }
}
```

> **Verify against the compiler:** `conn.remote_id()`, `conn.accept_bi()/open_bi()`, `conn.close(code, reason)`, `RecvStream: AsyncRead`, `endpoint.connect(EndpointAddr, ALPN)`. Adjust names if 1.0 differs; the design's §3 reference is the guide. The `conn_local_id_string` seam is filled where the hub's own `Endpoint::id()` is in scope (the accept loop in P2); for the test, the hub passes its id when building `HelloResult` — refactor `gate_connection` to take `hub_id: String` if cleaner.

- [ ] **Step 2: Write the integration test `tests/remote_pairing.rs`**

Two endpoints in one process, relay disabled + local discovery so no external infra. Build both with `presets::Minimal` + `.discovery_local_network()` (verify the builder method name). The hub spawns an accept task running `gate_connection`; the node dials.

```rust
//! P1 integration: pairing + HELLO handshake over a local (relay-disabled) iroh link.

use higgs::auth::{Allowlist, PairingTokens};
use higgs::node::{dial_and_hello, gate_connection, GateOutcome};

// Helper: bind a local-only endpoint (no relay) for in-process testing.
async fn local_endpoint() -> iroh::Endpoint {
    iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![higgs::remote::ALPN.to_vec()])
        .discovery_local_network()
        .relay_mode(iroh::RelayMode::Disabled)
        .bind()
        .await
        .expect("bind local endpoint")
}

#[tokio::test]
async fn valid_token_pairs_and_strangers_are_rejected() {
    let hub = local_endpoint().await;
    let node = local_endpoint().await;
    let hub_addr = hub.addr(); // EndpointAddr of the hub — verify accessor name

    // Hub: accept one connection, gate it with a freshly minted token.
    let mut allow = Allowlist::load(&std::env::temp_dir().join("p1-allow.json")).unwrap();
    let mut tokens = PairingTokens::new();
    let tok = tokens.mint(1_000, 600_000);

    let hub_task = tokio::spawn(async move {
        let incoming = hub.accept().await.expect("incoming");
        let conn = incoming.await.expect("conn");
        gate_connection(&conn, &mut allow, &mut tokens, 2_000, Some("studio".into())).await
    });

    let node_id = node.id().to_string();
    let res = dial_and_hello(&node, hub_addr, node_id, Some(tok)).await;
    assert!(res.is_ok(), "valid token should pair: {res:?}");

    let outcome = hub_task.await.unwrap();
    assert!(matches!(outcome, GateOutcome::Admitted { agreed_version: 1 }));
}
```

> Add two more `#[tokio::test]`s mirroring this shape: (a) **stranger** — no token, empty allowlist → `dial_and_hello` errs and the hub returns `Rejected` (HG024); (b) **stalled** — the node opens the bi-stream but never writes HELLO → after `HELLO_DEADLINE` the hub returns `Rejected` (HG028). Use `tokio::time` with a real (short) sleep for the stalled case, or temporarily lower `HELLO_DEADLINE` via a test-only setter. Verify endpoint accessor names (`endpoint.addr()` / `endpoint.id()`) against the compiler in Step 1.

- [ ] **Step 3: Build + run the integration test**

Run: `cargo test --test remote_pairing 2>&1 | tail -20` — Expected: PASS (all three cases). Network timing can be flaky; if the stalled test is slow, gate it behind a short deadline.

- [ ] **Step 4: Full gate + codex review loop + commit**

Run: `scripts/coverage.sh 2>&1 | tail -8` — Expected: ≥ 90%. (If the new network code dips coverage, the integration test must exercise both reject paths.)
`codex review --uncommitted` → converge. Then:

```bash
git add src/node/ tests/remote_pairing.rs
git commit -m "feat(node): post-HELLO gate + stalled-handshake deadline + dial; pairing integration test"
```

---

## Task 7: Minimal CLI — `--node`, `link pair`, `link status`, `node connect`

Just enough CLI to drive P1 by hand. Full fleet CLI is P6.

**Files:**
- Modify: `src/bin/higgs.rs`

- [ ] **Step 1: Add the role/subcommand arms**

In `main()` of `src/bin/higgs.rs`, after the `--higgs-worker` check, add a parse of the first non-flag arg. Keep it dependency-free (no clap) to match the existing hand-rolled style:

```rust
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("--node") => return higgs::node::run_node_cli(&args[1..]),
        Some("link") => return higgs::node::run_link_cli(&args[1..]),
        Some("node") => return higgs::node::run_node_subcmd(&args[1..]),
        _ => {} // fall through to the default server
    }
```

- [ ] **Step 2: Implement the CLI entrypoints in `src/node/mod.rs`**

Add `run_node_cli` (bind endpoint, print our `EndpointId`, run the accept loop calling `gate_connection`), `run_link_cli` (`pair` → mint a token + print it with the hub `EndpointAddr` as an `EndpointTicket` string; `status` → print our id + paired count), and `run_node_subcmd` (`connect <ticket>` → parse `EndpointTicket::from_str`, `dial_and_hello`). Each builds its own tokio runtime (`Runtime::new()`, multi-thread) since the bin's `main` is sync. Persist via `home::ensure_home()` + `Allowlist::load(home.join("pairings.json"))` + `load_or_create_secret(home.join("endpoint.key"))`.

> Keep these thin — they wire already-tested pieces. Print human output to stdout; errors to stderr with the `[HGxxx]` code. The `EndpointTicket` build/parse uses `iroh_tickets::endpoint::EndpointTicket` + `EndpointAddr::new(id).with_relay_url(..)` (verify accessors).

- [ ] **Step 3: Manual smoke (two terminals)**

```bash
# terminal A (hub)
cargo run --bin higgs -- link pair        # prints a ticket string + htk_ token
# terminal B (node)
cargo run --bin higgs -- node connect <ticket-from-A>   # should HELLO + pair
cargo run --bin higgs -- link status      # paired count = 1
```

Document the observed output in the commit message. (This is manual; no automated assertion.)

- [ ] **Step 4: Build + full gate + codex review loop + commit**

Run: `cargo build 2>&1 | tail -5 && scripts/coverage.sh 2>&1 | tail -6` — Expected: build clean, ≥ 90%.
`codex review --uncommitted` → converge. Then:

```bash
git add src/bin/higgs.rs src/node/
git commit -m "feat(cli): --node, link pair/status, node connect (P1 hand-drive)"
```

---

## Task 8: P1 acceptance

- [ ] **Step 1: Full suite + coverage**

Run: `scripts/coverage.sh 2>&1 | tail -20` — Expected: ≥ 90%, all tests pass.

- [ ] **Step 2: Full clippy**

Run: `cargo clippy --all-targets 2>&1 | tail -15` — Expected: no warnings.

- [ ] **Step 3: Confirm exit criteria (DESIGN-remote.md §10-P1)**

- two binaries pair (integration test + manual smoke) ✓
- HELLO agrees a version ✓
- stranger rejected post-HELLO (HG024) ✓
- silent post-QUIC peer dropped after the deadline (HG028) ✓
- version mismatch → typed close (HG023) ✓ (add a 4th integration case if not already covered)

- [ ] **Step 4: Cumulative codex review vs P0 tip + update roadmap**

`codex review --base <P0-tip-sha>` → converge. Update the roadmap ledger P1 → **DONE**, then:

```bash
git add docs/superpowers/plans/2026-06-19-iroh-remote-roadmap.md
git commit -m "docs(plan): P1 pairing+handshake complete; roadmap updated"
```

---

## Self-Review (against DESIGN-remote.md §3, §4.1, §7, §10-P1)

- **Coverage:** §3.1 SecretKey persistence → Task 1. §7 allowlist + pairing token → Tasks 2–3. §4.1 HELLO + negotiation → Task 4. §7.1 HG022/023/024/028 → Task 5. §3.2/§3.2.1 post-HELLO gate + stalled deadline → Task 6. §8 CLI (`link`/`node`) → Task 7.
- **Deferred (correct per spec):** chat/data streams, `NodeRuntime`, inventory, `M_*` lifecycle, ts-rs exports — P2+. `EndpointTicket` QR — P6 (P1 prints the plain ticket string).
- **Placeholders:** the iroh API accessor names (`sk.public()`, `endpoint.addr()`, `endpoint.id()`, `conn.remote_id()`, `discovery_local_network`, `relay_mode`) are flagged for compiler verification in Tasks 1 and 6 — they are the design's verified names but MUST compile; that's the point of Task 1 being the gateway.
- **Type consistency:** `HelloParams`/`HelloResult`/`negotiate_version`/`VersionMismatch` defined in Task 4 are used identically in Task 6. `Allowlist`/`PairingTokens`/`TokenError` from Tasks 2–3 used identically in Task 6. `GateOutcome` defined and asserted consistently in Task 6.
- **Crate boundary:** `auth.rs`/`remote.rs`/`node/` use only the crate's own serde + iroh; no common/engine/jigglebot import (§7 fix #10).
