# higgs-macros — design

## Why a separate crate

Proc-macros must compile as their own `proc-macro = true` crate. `higgs-macros`
is a workspace member (`higgs/Cargo.toml` → `[workspace] members = ["higgs-macros"]`)
that `higgs` depends on by path. Keeping it minimal (one derive, no runtime deps
beyond `syn`/`quote`/`proc-macro2`) keeps higgs's build + release self-contained.

## TsConstEnum: how it composes with ts-rs

`ts_rs::TS` is the single source of truth for a type's existence + import path.
For an enum we want BOTH:

- ts-rs to KNOW the type (so dependent structs emit `import type { X } from "./X"`),
- but the emitted `X.ts` to be a const-object, not a union.

So the enum derives `ts_rs::TS` **without** `#[ts(export)]` (ts-rs won't emit its
own `export_bindings_*` writer for it) and adds `TsConstEnum`, which emits its own
`macro_run_*` writer. The enum keeps `#[ts(export_to = "...")]` so both ts-rs (for
the import path of dependents) and TsConstEnum (for where to write) agree on the
location.

### The transitive-export gotcha (why ordering matters)

ts-rs's `T::export()` for a struct `T` with `#[ts(export)]` writes `T.ts` **and
its transitive dependencies**, using ts-rs's default union form. So a struct that
references `FlashAttn` re-writes `FlashAttn.ts` as a union as a side effect — even
though `FlashAttn` itself has no `#[ts(export)]`. The fix is purely **ordering**:
run the normal ts-rs pass first, then the `macro_run` pass last so the const-object
overwrites the union. `scripts/quality.sh` enforces this order; the bindings-sync
check then proves the committed `.ts` matches the regen.

## Path anchoring

ts-rs anchors `export_to` at `<crate>/bindings/`; the `macro_run` writer anchors at
`CARGO_MANIFEST_DIR` (`<crate>/`). higgs's `export_to` is `bindings/`-relative
(`"higgs/"` → `<crate>/bindings/higgs/`), so `reanchor_ts_path` prepends `bindings/`.
(Jigglebot's original macro instead stripped a leading `../`, because its paths are
`../../../frontend/...`-style — the only behavioral difference in this port.)

## Constraints

- Unit-variant enums only — variants with fields are a compile error (a
  discriminated union can't be a const-object; those stay on `higgs_ts!`).
- Honors `#[serde(rename_all)]` and per-variant `#[serde(rename)]` for wire values.
