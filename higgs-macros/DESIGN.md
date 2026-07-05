# higgs-macros — design

## Why a separate crate

Proc-macros must compile as their own `proc-macro = true` crate (`Cargo.toml`).
`higgs-macros` is a workspace member that `higgs` depends on by path. It is kept
deliberately minimal — two derives, no runtime deps beyond `syn` / `quote` /
`proc-macro2` — so higgs's build and release stay self-contained. The derive was
ported from jigglebot's `macros` crate (higgs is standalone and can't depend on
it); keeping a local copy keeps higgs's enum bindings byte-consistent with
jigglebot's.

Everything is compile-time codegen. **This module owns no `HGxxx` runtime error
codes** — its only failure mode is a `syn::Error` compile error (surfaced via
`to_compile_error()` in `derive_ts_const_enum` / `derive_ts_param_help`). There is
no concurrency or locking model; the only ordering concern is the ts-rs-vs-writer
sequence described below, which is a build-script/test ordering issue, not runtime.

## TsConstEnum: how it composes with ts-rs

`ts_rs::TS` is the single source of truth for a type's existence + import path. For
an enum we want BOTH:

- ts-rs to KNOW the type (so dependent structs emit `import type { X } from "./X"`),
- but the emitted `X.ts` to be a const-object, not a union.

So each enum (via the `higgs_const_enum!` wrapper in higgs's `src/ts_export.rs`)
derives `ts_rs::TS` **without** `#[ts(export)]` — ts-rs won't emit its own
`export_bindings_*` writer for it — and adds `TsConstEnum`, whose `expand`
generates a private `macro_run_*` writer test instead. The enum keeps
`#[ts(export_to = "…")]` so both ts-rs (for a dependent's import path) and
`TsConstEnum` (for where to write) agree on one location declared once. `expand`
reads that attribute back via `find_ts_export_to`.

### The transitive-export gotcha (why ordering matters)

ts-rs's `T::export()` for a struct `T` with `#[ts(export)]` writes `T.ts` **and its
transitive dependencies** in ts-rs's default union form. So a struct that
references `FlashAttn` re-writes `FlashAttn.ts` as a union as a side effect — even
though `FlashAttn` itself has no `#[ts(export)]`. The fix is purely **ordering**:
run the normal ts-rs pass first, then the `macro_run` pass (`cargo test macro_run
-- --ignored`) last so the const-object overwrites the union. Both writer tests are
`#[ignore]`d precisely so a normal `cargo test` never fires them while ts-rs's
auto-export writer is active — the `--ignored` pass runs them alone, and the single
`fs::write` wins deterministically. `scripts/quality.sh` enforces this order, then
the bindings-sync check proves the committed `.ts` matches the regen.

## TsParamHelp: why a second, different emitter

ts-rs already carries doc comments into `.ts` as JSDoc, but JSDoc is compile-time
metadata — it cannot be read at runtime to populate a UI tooltip. `TsParamHelp`
(`expand_param_help`) therefore emits a real **data table** the frontend can index:

- a Rust `pub const PARAM_HELP: &[(&str, &str)]` on the type (so the consuming crate
  can unit-test the annotations — e.g. `FitVerdict::PARAM_HELP` in
  `src/tune/mod.rs`), and
- a `<Type>Help.ts` const-object keyed by wire name.

It reuses the same `#[ts(export_to)]` so the help map lands beside the type's own
binding, and reuses the same `#[ignore]d macro_run_*` writer discipline. `#[help]`
is declared as an inert `attributes(help)` helper so it never reaches serde/ts-rs;
the wire format is untouched.

### Keys are serde WIRE names, so they must be quoted

`TsConstEnum` keys are Rust variant idents, always valid TS identifiers, so it emits
them bare. `TsParamHelp` keys are serde **wire names** (`find_serde_rename`, else the
container `rename_all` rule) — a rename like `#[serde(rename = "max-tokens")]` is not
a valid TS identifier. `render_help_ts` therefore emits **both key and value as
escaped double-quoted string literals** (`"max-tokens": "…"`), which is always valid
and still permits dot access (`Help.ctx_len`) for identifier keys. `escape_ts_string`
handles every char a single-line `"…"` literal can't carry raw: backslash,
double-quote, the C0 control range, and the JS line separators U+2028/U+2029 — so the
generated file is valid for ANY help prose, not just today's strings.

## Serde-matching case conversion (the subtle part)

Both derives must reproduce the EXACT wire name serde/ts-rs would serialize, so the
const-object values and the help-map keys line up with the actual JSON. The
`rename_all` rules are applied by hand because serde's own logic isn't reusable from
a proc-macro. Crucially, serde applies a `rename_all` rule **differently to enum
variants vs struct fields** (different source-case assumptions), so there are
parallel helper families:

- **Variants** (`PascalCase` source): `apply_rename_all` (used by `TsConstEnum`) and
  `apply_rename_all_variant` (used by `TsParamHelp` enums), backed by `pascal_to_snake`.
- **Fields** (`snake_case` source): `apply_rename_all_field`, backed by
  `field_to_pascal`.

Getting the source-case wrong (e.g. running the variant rule on a field) yields a
wire name that silently diverges from serde's, which the bindings-sync gate would
eventually catch — but the split exists so it never gets that far. `to_snake_case` /
`to_camel_case` are shared low-level helpers (also used to name the `macro_run_*`
test functions).

## Path anchoring

ts-rs anchors `export_to` at `<crate>/bindings/`; the `macro_run` writer anchors at
`CARGO_MANIFEST_DIR` (`<crate>/`, resolved at the *consuming* crate's compile time,
which is why the files land in `higgs/bindings/higgs/`, not under `higgs-macros/`).
higgs's `export_to` is `bindings/`-relative (`"higgs/"` → `<crate>/bindings/higgs/`),
so `reanchor_ts_path` simply prepends `bindings/` to resolve to the same final
directory. (Jigglebot's original macro instead stripped a leading `../` because its
paths are `../../../frontend/…`-style — the only behavioral difference in this port.)

## Constraints & invariants

- `TsConstEnum`: enums only, **unit variants only** — variants with fields are a
  compile error (`expand` rejects non-`Fields::Unit`); a discriminated union can't be
  a const-object, so those stay on higgs's `higgs_ts!` wrapper.
- `TsParamHelp`: **named-field structs** or **unit-variant enums** only; tuple/unit
  structs, enum variants with fields, and unions are compile errors. Fields/variants
  without `#[help]` are simply skipped (not every field needs a tooltip).
- Both require `#[ts(export_to = "…")]` and error clearly if it is missing (so ts-rs
  and the writer can never disagree on the output path).
- `#[serde(rename_all)]` (container) and `#[serde(rename)]` (per field/variant) are
  honored for wire values/keys; a per-item `rename` always wins over the container
  rule.
- `PARAM_HELP` is emitted with **elided reference lifetimes** (not explicit
  `'static`) so clippy's `redundant_static_lifetimes` stays quiet under the crate's
  `-D warnings` gate.

## Deferred / residual

- `apply_rename_all` / `apply_rename_all_field` fall back to identity for an
  unrecognized `rename_all` rule, whereas `apply_rename_all_variant` `panic!`s on an
  unsupported rule. This asymmetry is intentional-but-uneven; no higgs type currently
  exercises an unusual rule, so it hasn't been unified.
- The attribute parsers walk `#[ts]`/`#[serde]` meta by hand (`parse_nested_meta`)
  and ignore unknown keys rather than depending on ts-rs's or serde's internal
  parsers — a deliberate decoupling that keeps this crate's dep surface tiny at the
  cost of re-implementing a little of their attribute grammar.
