//! Shared control helpers the [`Higgs`] facade delegates to.
//!
//! higgs no longer serves a `/api/higgs/*` HTTP surface — control runs in-process
//! via the crate API (`src/api/embed.rs`). What remains here are the PURE
//! sub-primitives that assembly used to share with those handlers and that the
//! facade methods now call directly: per-model row formatting ([`model_entry`] +
//! the tune-view plumbing), the hub-status snapshot ([`hub_status`]), and the
//! mint/revoke decision cores ([`decide_mint`] / [`decide_revoke`] +
//! [`validate_key_label`] / [`keystore_io_error`]). One implementation, reached by
//! its `pub(crate)` path from the facade.

use super::wire::{HiggsHubStatus, HiggsModelEntry};
use crate::api::Higgs;
use crate::diagnostic::HiggsError;
use crate::worker::engine::LoadParams;
use crate::worker::models::HiggsModel;

/// Gate 2 (host-side, zero FFI): does this model's chat template declare tool
/// handling? llama.cpp's auto-parser derives the actual parser from the
/// template at load time, so the scan-time signal is the template mentioning
/// tools/functions (same heuristic as the scan's `supports_tools`); `false`
/// when there is no template (the legacy route renders no tool grammar).
fn tool_calls_supported(model: &HiggsModel) -> bool {
    match model.chat_template.as_deref() {
        Some(tmpl) => {
            let tl = tmpl.to_lowercase();
            tl.contains("tool") || tl.contains("function")
        }
        None => false,
    }
}

/// Build the per-model control entry: the canonical [`HiggsModel`] enriched with
/// its load `state`, `format`, the Gate-2 tool-call support verdict, and the
/// `last_load` params persisted on the last successful load (if any).
///
/// There is NO load-to-test probe at scan time — engine loadability is learned
/// only when the model is actually loaded (the load error is surfaced then).
/// `support_reason` carries the fixed Gate-2 message when no tool-call parser
/// matches the model's template, else `None`.
pub(crate) fn model_entry(
    mut model: HiggsModel,
    loaded_ids: &[String],
    last_load: Option<LoadParams>,
    readiness: crate::serve::readiness::ModelReadiness,
    fit: Option<crate::serve::wire::ModelFit>,
    tune: TuneProfileViews<'_>,
) -> HiggsModelEntry {
    // Multi-model: this model is "loaded" if it is among the resident ids, not only
    // when it is the primary.
    let is_loaded = loaded_ids.iter().any(|id| id == &model.id);
    let tool_calls = tool_calls_supported(&model);
    // Gate 2 (pure host-side template sniff): no parser matches the template.
    let support_reason =
        (!tool_calls).then(|| "no tool-call parser matches this model's template".to_owned());
    // The transient chat_template never leaves the host; drop it explicitly
    // (it is `serde(skip)` anyway). `gguf_components` stays on the model — its
    // single home — and rides the flattened payload.
    model.chat_template = None;
    HiggsModelEntry {
        state: if is_loaded {
            "loaded".to_owned()
        } else {
            "not-loaded".to_owned()
        },
        format: "gguf".to_owned(),
        tool_calls,
        support_reason,
        last_load,
        readiness,
        fit,
        tuned_load: tune.analytical.map(|t| t.profile.clone()),
        benched_load: tune.bench.map(|t| t.profile.clone()),
        tune_provenance: tune.active.map(|t| t.provenance),
        bench_tps: tune.bench.and_then(|t| t.bench_tps),
        model,
    }
}

/// The per-model tune views the entry serves: the ACTIVE record (JIT default,
/// its provenance labels the default set) plus the analytical/bench history
/// slots. `from_triple` grandfathers a pre-dual-slot store — one whose BOTH
/// history slots are empty (an old `models.json` predating the slots) — by
/// serving its lone active record under its own provenance's label. Once
/// EITHER slot is populated the store is post-dual (`put_tuning` fills a slot
/// on every real tune/turbotune), so the active record is NOT borrowed: a later
/// bare load can demote it to a `Heuristic` that was never an analytical tune,
/// and borrowing that would fabricate a "Tuned" set. (Residual A: on a store
/// whose only write was a bare load, that lone record is indistinguishable from
/// a pre-dual analytical tune and is served as Tuned — a real tune supersedes it.
/// Residual B: on a pre-dual store whose FIRST post-upgrade action is a benchmark,
/// the bench slot populates and the gate turns off, so the prior analytical tune
/// stops being grandfathered as "Tuned" until the user re-runs Tune — accepted
/// because that analytical set is deterministic and re-derivable in one click,
/// whereas `put_tuning` backfilling the active record to keep it would reintroduce
/// the bare-load masquerade this gate exists to prevent.)
#[derive(Clone, Copy, Default)]
pub(crate) struct TuneProfileViews<'a> {
    active: Option<&'a crate::tune::store::TuneRecord>,
    analytical: Option<&'a crate::tune::store::TuneRecord>,
    bench: Option<&'a crate::tune::store::TuneRecord>,
}

pub(crate) type TuneTriple = (
    Option<crate::tune::store::TuneRecord>,
    Option<crate::tune::store::TuneRecord>,
    Option<crate::tune::store::TuneRecord>,
);

/// The ACTIVE ("latest") tuning record per model, extracted from the
/// `(active, analytical, bench)` triples the models list ALSO reads for its
/// dual-profile wire fields. Readiness reads THIS map, so readiness and the
/// `tuned_load`/`benched_load` fields both come from ONE `models.json` snapshot —
/// a concurrent tune between two separate store reads can no longer make a row's
/// readiness disagree with its profile fields.
pub(crate) fn active_records(
    profiles: &std::collections::BTreeMap<String, TuneTriple>,
) -> std::collections::BTreeMap<String, crate::tune::store::TuneRecord> {
    profiles
        .iter()
        .filter_map(|(id, (active, _, _))| active.clone().map(|a| (id.clone(), a)))
        .collect()
}

impl<'a> TuneProfileViews<'a> {
    pub(crate) fn from_triple(triple: Option<&'a TuneTriple>) -> Self {
        let Some((active, analytical, bench)) = triple else {
            return Self::default();
        };
        let active = active.as_ref();
        let mut analytical = analytical.as_ref();
        let mut bench = bench.as_ref();
        // Grandfather ONLY a true pre-dual store (both history slots empty). A
        // post-dual store always has a slot filled by `put_tuning`, so borrowing
        // the active record there would surface a bare-load `Heuristic` — a manual
        // reload that `set_profile` demoted after a turbotune — as the analytical
        // "Tuned" set it never was.
        if analytical.is_none() && bench.is_none() {
            if let Some(a) = active {
                if a.provenance == crate::tune::TuneProvenance::Bench {
                    bench = Some(a);
                } else {
                    analytical = Some(a);
                }
            }
        }
        Self {
            active,
            analytical,
            bench,
        }
    }
}

/// Build the current hub-mode status. `enabled` = the hub network is up (accepting dials);
/// `node_count` is the fleet size, which persists across a disable (nodes then show disconnected
/// until the hub is re-enabled and they reconnect).
pub(crate) async fn hub_status(higgs: &Higgs) -> HiggsHubStatus {
    let node_count = match higgs.fleet() {
        Some(fleet) => u32::try_from(fleet.nodes_view().await.len()).unwrap_or(u32::MAX),
        None => 0,
    };
    match higgs.hub() {
        Some(hub) => HiggsHubStatus {
            enabled: true,
            hub_id: Some(hub.hub_id().to_string()),
            node_count,
        },
        None => HiggsHubStatus {
            enabled: false,
            hub_id: None,
            node_count,
        },
    }
}

// ── API-key management (G4): the mint / revoke DECISION cores ───────────────
//
// The bearer-authenticated mutation glue lives on the facade
// (`Higgs::mint_key` / `Higgs::revoke_key`); these pure decision functions are
// shared so the in-process and (historically) HTTP callers enforce the same
// invariants over the LOCKED keystore.

/// The outcome of a mint decision, computed against the LOCKED keystore.
pub(crate) enum Mint {
    /// Mint with these effective scopes.
    Ok(Vec<crate::keys::Scope>),
    /// A key with this label already exists.
    Duplicate,
    /// Non-bootstrap mint with no valid Admin bearer for the current store.
    Unauthorized,
    /// A BOOTSTRAP mint (first key) whose EXPLICIT scopes omit `admin`: it would
    /// flip auth on yet be unable to reach the Admin-scoped key-management API,
    /// with no HTTP path left to recover ([codex r10]).
    BootstrapNeedsAdmin,
}

/// Validate a mint label's charset. Labels address a key for revocation as a
/// SINGLE path segment — anything that can't round-trip through such a segment
/// (slashes, whitespace, control chars) must be rejected at mint or the key
/// becomes unrevokable. `.` / `..` pass the charset but URL parsers normalize
/// dot-segments away, so those are rejected too. Shared by the crate API's
/// [`Higgs::mint_key`](crate::api::Higgs::mint_key).
pub(crate) fn validate_key_label(label: &str) -> Result<(), HiggsError> {
    let label_ok = !label.is_empty()
        && label.len() <= 64
        && label != "."
        && label != ".."
        && label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if label_ok {
        Ok(())
    } else {
        Err(HiggsError::InvalidKeyRequest {
            detail: format!(
                "invalid key label {label:?}: 1-64 chars from [A-Za-z0-9._-] (it must fit in a single URL path segment)"
            ),
        })
    }
}

/// Decide a mint against the CURRENT (locked) keystore `ks`. Pure — derives
/// `bootstrap` from `ks` itself, so the empty-store window can't be raced: a
/// second unauthenticated mint that reaches the lock after the first key
/// landed sees a non-empty store, fails the bearer recheck, and is refused.
/// The bootstrap mint (empty store) is allowed unauthenticated and MUST grant
/// `Admin` — it defaults to `[admin]` when scopes are omitted, and REJECTS
/// explicit scopes that omit `admin` ([`Mint::BootstrapNeedsAdmin`]), since a
/// non-admin first key would flip auth on and lock the HTTP management surface
/// out of itself with no recovery path. Later omitted-scopes mints default to
/// `[chat, models]`. Explicit `requested` scopes otherwise win.
///
/// `trusted` = an IN-PROCESS caller (the crate API's [`Higgs::mint_key`]): it
/// short-circuits ONLY the bearer [`Mint::Unauthorized`] branch — EVERY structural
/// invariant below (duplicate, bootstrap-needs-admin, scope defaults) still runs.
pub(crate) fn decide_mint(
    ks: &crate::keys::ApiKeys,
    trusted: bool,
    bearer: Option<&str>,
    requested: Option<Vec<crate::keys::Scope>>,
    label: &str,
) -> Mint {
    use crate::keys::Scope;
    // `bootstrap` = the WHOLE store is empty (hidden keys included). Deliberately
    // NOT `visible().next().is_none()`: with a hidden internal Admin present
    // (embedded mode) the store is already auth-enabled and manageable via that
    // hidden bearer, so the "first key must be Admin so you don't lock yourself
    // out" rule does not apply — and forcing the first VISIBLE key to Admin would
    // OVER-GRANT (an external-app token would silently become Admin). A caller who
    // wants an admin key still requests `Admin` explicitly; standalone `higgs keys
    // add` can always mint one directly. Auth stays gated on the full store below.
    let bootstrap = ks.is_empty();
    if !trusted && !bootstrap && !bearer.is_some_and(|t| ks.authorizes(t, Scope::Admin)) {
        return Mint::Unauthorized;
    }
    if ks.iter().any(|k| k.label == label) {
        return Mint::Duplicate;
    }
    // The FIRST key must be able to manage keys — reject explicit non-admin
    // bootstrap scopes rather than mint a self-locking key.
    if bootstrap {
        if let Some(scopes) = &requested {
            if !scopes.contains(&Scope::Admin) {
                return Mint::BootstrapNeedsAdmin;
            }
        }
    }
    let scopes = requested.unwrap_or_else(|| {
        if bootstrap {
            vec![Scope::Admin]
        } else {
            vec![Scope::Chat, Scope::Models]
        }
    });
    Mint::Ok(scopes)
}

/// The outcome of a revoke decision, computed against the LOCKED keystore.
pub(crate) enum Revoke {
    /// Remove keys with the label; carries the count that will be removed.
    Removed(usize),
    /// The store became non-empty (a bootstrap mint won a race) and the request
    /// carries no Admin bearer — refuse (codex r7, mirrors the mint recheck).
    Unauthorized,
    /// Would empty the keystore while LAN-exposed ([HG059]).
    LastKeyOnLan,
    /// Would remove the LAST Admin-capable key while OTHER (non-Admin) keys remain
    /// — auth stays on but the Admin-only management surface becomes unreachable, a
    /// lockout ([HG066]). Emptying the store entirely (turning auth off) is allowed;
    /// stranding non-Admin keys is not.
    LastAdminKey,
}

/// Decide a revoke against the CURRENT (locked) keystore `ks`. Pure. A non-empty
/// store REQUIRES a live Admin bearer: authorization is re-derived from the
/// locked store, not trusted from an earlier pass, so a DELETE admitted while the
/// store was empty (auth off) is refused if a concurrent bootstrap mint committed
/// first. `lan_exposed` gates the last-key [HG059] refusal.
///
/// `trusted` = an IN-PROCESS caller (the crate API's [`Higgs::revoke_key`]): it
/// short-circuits ONLY the bearer [`Revoke::Unauthorized`] branch — the last-admin
/// ([HG066]) and last-key-on-LAN ([HG059]) invariants below still run.
pub(crate) fn decide_revoke(
    ks: &crate::keys::ApiKeys,
    trusted: bool,
    bearer: Option<&str>,
    label: &str,
    lan_exposed: bool,
) -> Revoke {
    if !trusted
        && !ks.is_empty()
        && !bearer.is_some_and(|t| ks.authorizes(t, crate::keys::Scope::Admin))
    {
        return Revoke::Unauthorized;
    }
    // Every structural guard below operates over the VISIBLE keystore ONLY. A
    // hidden internal key (the embedder's in-memory token, [`ApiKeys::add_internal`])
    // is never persisted and never listable, so it must not influence a revoke
    // decision: counting it in `total` would make `matching < total` true even
    // when the user deletes their sole VISIBLE key, spuriously tripping the
    // last-admin guard; counting it in `admin_remains` would let the last VISIBLE
    // admin be revoked and strand the persisted management surface. The earlier
    // auth re-check (is_empty / authorizes) still uses the FULL store — that is an
    // AUTH question, and the hidden token legitimately authorizes. `visible() ==
    // iter()` for a standalone higgs (no hidden keys), so this is byte-identical
    // there.
    let total = ks.visible().count();
    let matching = ks.visible().filter(|k| k.label == label).count();
    if matching > 0 && lan_exposed && matching == total {
        return Revoke::LastKeyOnLan;
    }
    // Removing the LAST VISIBLE Admin key while OTHER visible keys remain locks the
    // Admin-only management surface out of the persisted store — refuse. Two guards
    // must both hold for that to be the case:
    //   • the revoke actually removes a visible Admin (`removing_an_admin`) — revoking
    //     a NON-admin key can't strand a management surface, even a store that never
    //     had a visible Admin (reachable when an embedder mints scoped keys under its
    //     hidden Admin), so those revokes stay allowed; and
    //   • no visible Admin would remain afterwards (`!admin_remains`).
    // Emptying the visible store entirely (matching == total → the persisted surface
    // turns auth OFF) is a separate, allowed operation, gated only by the LAN check.
    if matching > 0 && matching < total {
        let removing_an_admin = ks
            .visible()
            .filter(|k| k.label == label)
            .any(|k| k.scopes.contains(&crate::keys::Scope::Admin));
        let admin_remains = ks
            .visible()
            .filter(|k| k.label != label)
            .any(|k| k.scopes.contains(&crate::keys::Scope::Admin));
        if removing_an_admin && !admin_remains {
            return Revoke::LastAdminKey;
        }
    }
    Revoke::Removed(matching)
}

/// Map a keystore file I/O failure onto the coded store error ([HG040]).
pub(crate) fn keystore_io_error(e: std::io::Error) -> HiggsError {
    HiggsError::PersistenceFailed {
        store: "api_keys".into(),
        path: crate::keys::keys_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|_| "api_keys.json".into()),
        source: e,
    }
}

#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
