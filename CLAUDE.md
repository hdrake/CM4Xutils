# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`CM4Xutils` loads and postprocesses output from the CM4X coupled climate model simulations
(GFDL CM4 with MOM6 ocean at 1/4° = `CM4Xp25` and 1/8° = `CM4Xp125`). It is a thin,
opinionated layer that turns GFDL post-processed (`pp`) NetCDF diagnostics into
`xgcm.Grid` objects that `xwmt` / `xwmb` / `sectionate` / `regionate` can consume for
water mass transformation and budget analysis. See also
https://github.com/StephenGriffies/CM4X (the model documentation papers in JAMES).

Loading depends on GFDL infrastructure — `doralite` (the Dora experiment metadata
service), `gfdl_utils.core.open_frompp`, `/archive` paths, and `dmget` tape retrieval —
so the `load_*` functions only work on GFDL analysis machines. The coarsening,
remapping, and grid utilities are portable and can be exercised on the Zarr stores in
`data/coarsened/`.

## Development workflow

**All branch/PR work happens in an isolated git worktree.** The main checkout
(`/work5/hfd/codedev/CM4Xutils`) stays on `main`; never develop a feature or fix there.
Worktrees live in `.claude/worktrees/<name>/` (Claude Code's `EnterWorktree` puts them
there automatically), one per branch/PR:

```bash
git worktree add .claude/worktrees/<name> -b <branch>
```

**Each worktree gets its own conda environment**, named `CM4Xutils_<name>` after the
worktree, with *that worktree's* CM4Xutils installed editably into it:

```bash
conda create -n CM4Xutils_<name> --clone CM4Xutils   # or build from environment.yml
conda activate CM4Xutils_<name>
pip install -e /work5/hfd/codedev/CM4Xutils/.claude/worktrees/<name>
```

The per-worktree environment is not optional bookkeeping: branches routinely carry
incompatible dependency pins (`xgcm < 0.10` vs `>= 0.10`, `xbudget 0.5` vs `0.7`, a
git-pinned vs released `xwmt`), so a single shared environment would leave some branches
unrunnable and silently test others against the wrong stack.

**Always verify a branch with its own environment.** Plain `python` resolves to whatever
env is active — usually the base `CM4Xutils`, which tracks `main`'s dependency set — and
will import the *main checkout's* CM4Xutils, not the branch's. Confirm before trusting a
result:

```bash
python -c "import CM4Xutils, xgcm; print(CM4Xutils.__file__, xgcm.__version__)"
```

The path must point inside the worktree you are working in. Beware that running `python`
from inside a worktree puts the worktree on `sys.path` regardless of the active
environment, so a correct-looking import path does *not* by itself prove the right
environment is active — check a pinned dependency's version too.

## Commands

```bash
pip install -e .                       # editable install
pip install git+https://github.com/hdrake/CM4Xutils.git@main   # user install
```

There is no test suite and no CI. Validation is done by running the notebooks in
`examples/` (`wmb_from_zarr.ipynb` is the cheapest end-to-end check — it reads a local
Zarr store rather than `/archive`).

Production data generation (run from `scripts/`, requires GFDL filesystem):

```bash
cd scripts
python coarsen_sigma2_budgets.py CM4Xp25 1750 5            # model, interval_start, interval_length (years)
python coarsen_sigma2_tracers.py CM4Xp25 historical 1850 5 # model, experiment, interval_start, interval_length
bash dmget_CM4Xp25-historical_ocean_month_rho2.sh          # pre-stage tape files
```

Intervals are always multiples of 5 (the pp files are 5-year chunks). Outputs land in
`data/coarsened/{model}_budgets_sigma2_{YYYY}-{YYYY}.zarr`.

## Architecture

### Module layering

`CM4Xutils/__init__.py` star-imports everything, in dependency order:

- `grid_preprocess.py` — grid coordinate fixes, `ds_to_grid`, `cell_methods` parsing. No
  GFDL dependencies. Everything else imports from here.
- `transform.py` — `remap_vertical_coord`: conservative vertical remapping (z → sigma2).
- `coarsen.py` — `horizontally_coarsen`: grid-aware horizontal coarsening.
- `loading.py` — GFDL-specific loaders (`load_wmt_ds`, `load_wmt_grid`, `load_density`,
  `load_tracer`, `load_transient_tracers`) plus scenario/time alignment.

The canonical pipeline (see `scripts/remap_functions.py`) is:
`load_wmt_grid` → `add_sigma2_coords` → `remap_vertical_coord("sigma2", ...)` →
`ds_to_grid` → `horizontally_coarsen` → `to_zarr`.

### `cell_methods` is load-bearing metadata

This is the single most important convention in the codebase. Both
`horizontally_coarsen` and `remap_vertical_coord` dispatch on each variable's
`cell_methods` attribute (e.g. `"area:mean z_l:sum yh:mean xh:mean time:mean"`) to decide
whether to area-weight, volume-weight, sum, or subsample along each dimension:

- `{xh:mean, yh:mean}` → area-weighted average (or volume-weighted if `z_l:mean`, i.e. an
  intensive tracer like `thetao`; `wet` is a special case weighted by *total* cell area).
- `{xh:sum, yh:sum}` → wet-masked sum (`areacello`).
- `point` → subsample.
- Mixed cases (`wet_u`, `wet_v`) are handled per-axis, weighted by the face widths
  `dyCu` / `dxCv`.

A variable **without** a `cell_methods` attribute is silently skipped with a printed
warning. When adding or deriving a new variable anywhere in the codebase, set
`cell_methods` (and `cell_measures`) explicitly — see how `rsdoabsorb`,
`boundary_forcing_h_tendency`, and the interpolated `taux`/`tauy` do it in `loading.py`.
`parse_cell_methods` / `stringify_cell_methods_dict` round-trip the string form.

### Grid coordinates

CM4X static files have **wrong** `geolon`/`geolat`. The authoritative grid is the
supergrid ("hgrid") file, hard-coded per model in `exp_dict[model]["hgrid"]`.
`fix_geo_coords(og, sg)` reconstructs all eight geo-coordinate arrays from the supergrid
by strided slicing, auto-detecting whether the static file is native (`nx//2`) or already
d2-coarsened (`nx//4`).

`add_grid_coords(ds, og)` then attaches those coords to a diagnostics dataset and
**replaces `xh`/`yh`/`xq`/`yq` with plain integer indices** — nominal lat/lon are dropped
in favor of `geolat`/`geolon` 2D coords. Downstream code assumes this.

`ds_to_grid(ds, Zprefix=None)` builds the `xgcm.Grid` with X/Y (`xh`/`xq`, `yh`/`yq`),
an inferred Z axis, `areacello` as the (X,Y) metric, `boundary={"X":"periodic",
"Y":"extend", "Z":"extend"}`, and `autoparse_metadata=False`. Z inference order:
`sigma2_l` → `rho2_l` → `zl` → `z_l`; pass `Zprefix="z_"` to force it.

### The CM4Xp125 "d2" complication

For `CM4Xp125`, 3D budget tendencies are only archived on a 2×-coarsened ("d2") grid
while surface fluxes are archived on the native grid. `load_wmt_averages_and_snapshots`
therefore coarsens the surface fields by `{X:2, Y:2}` at load time. Additionally the
archived d2 static file coordinates are wrong, so `make_wmt_grid` regenerates them by
coarsening the *full-resolution* static file, and derives a 3D `wet_mask` from native
`thkcello` which it applies to every mean-mean-sum variable. Sea ice diagnostics come on
their own grid (`xT`/`yT` → `xh_ice`/`yh_ice`) and are coarsened/merged by `regrid_ice`.
Anything touching CM4Xp125 grid metadata needs to respect this whole chain.

### Experiments, time axes, and the `exp` dimension

`exp_dict` maps `(model, experiment)` → Dora `odiv-` IDs; `pp_dict` holds hard-coded
`/archive` paths used as a fallback when Dora is down (`get_wmt_pathDict` try/excepts on
this). Experiments: `piControl-spinup`, `piControl`, `piControl-continued`, `historical`,
`ssp585`.

Datasets carry **two time dimensions**: `time` (5-year-chunked monthly means) and
`time_bounds` (instantaneous snapshots). Snapshot variables are renamed with a `_bounds`
suffix (`thetao_bounds`, `sigma2_bounds`, …) so both can live in one dataset.
Loading a single 5-year interval also pulls the *last snapshot of the preceding
interval*, with hard-coded corner cases at experiment branch points (`0096`, `1845`,
`0356` for p25, `0446` for p125, `2010`) where the preceding interval belongs to a
different experiment.

`load_wmt_ds` concatenates the control and forced branches into an `exp` dimension with
values `["control", "forced"]`. `align_dates` maps control years onto historical years
(offset +1749) and stores the original control calendar as `time_since_init` /
`time_bounds_since_init`.

### Vertical remapping

`remap_vertical_coord("sigma2", ds, grid)` uses `xgcm`'s conservative transform. Extensive
variables (`z_l:sum`) are remapped directly; intensive ones (`z_l:mean`) are multiplied by
`thkcello`, remapped, then divided by the remapped thickness. The target
sigma2 grid is the 74-layer coordinate in `data/sigma2_coords.nc`, which
`add_sigma2_coords` loads and pads with one extra layer at each end to cover all
plausible ocean densities.

**Transports come from the native rho2 diagnostics where available (v1.4.0+).** MOM6
accumulates the layer-integrated mass transports `umo`/`vmo` *online* into
potential-density (`rho2`) layers in the `ocean_month_rho2` pp output, where they conserve
mass exactly within each layer. The budget pipeline sources them from there:
`load_rho2_transports` / `load_rho2_transports_ds` (loading.py) load the native
transports, and `rho2_transports_to_sigma2` (grid_preprocess.py) relabels `rho2_l →
sigma2_l` (`sigma2 = rho2 − 1000`) and zero-pads the two expansion layers onto the
76-layer sigma2 grid — no offline vertical remapping. Because `ocean_month_rho2` is
archived at *native* resolution (even for CM4Xp125, whose budget tendencies are on d2),
the transports are coarsened by `{X:12, Y:10}` for CM4Xp125 (vs `{X:6, Y:5}` for the d2
budgets) so they land on the same coarse grid. `remap_functions.py` then adopts the budget
product's horizontal coordinates onto the coarsened transports before merging them in.

**Not every model archives both transports.** CM4Xp125 saves both `umo` and `vmo` in
`ocean_month_rho2`, but **CM4Xp25 saves only `vmo`** (no `umo`). `available_rho2_transports`
/ `native_rho2_transport_vars` (loading.py) probe the filesystem, and
`remap_budgets_to_sigma2_and_coarsen` only uses the native path when **both** `umo` and
`vmo` are present across every experiment in the interval (i.e. CM4Xp125). Otherwise
(CM4Xp25) it falls back to the older offline z→sigma2 transport remap for both terms — via
`itp_tracer_to_transports` (the v1.2.0 fix, returning NaN where either neighbor is dry),
preserved behind `remap_vertical_coord(..., remap_transports=True)` (the default). So
CM4Xp25 `umo`/`vmo` are unchanged from v1.3.0; only CM4Xp125 transports change in v1.4.0.

## Versioning and provenance

`CM4Xutils/version.py` holds the package version; it is stamped into the `provenance`
attribute of every coarsened/remapped dataset. The generation scripts *separately* stamp
`ds.attrs["version"]` and `ds.attrs["version_notes"]` describing the dataset release.

When a change alters numerical output (which most bug fixes here do — see the git log:
d2 coarsening, area masks, boundary conditions, transport interpolation), bump the
package version, update the dataset `version`/`version_notes` strings in
`scripts/coarsen_sigma2_*.py`, and regenerate. Older buggy outputs are kept in parallel
directories (`data/coarsened_d2_bug/`, `data/coarsened_incorrect_wetmask/`,
`data/coarsened_nanbugged/`, …) for comparison — don't delete these.

`data/` is not git-tracked apart from `sigma2_coords.nc`; most of it is generated or
staged output.

## Known rough edges

- Datasets in `data/coarsened/` predate v1.4.0: they were written without the `thkcello`
  that `add_grid_coords` derives from `volcello/areacello`, and with `umo`/`vmo` derived by
  the old offline z→sigma2 remap rather than the native `ocean_month_rho2` diagnostics
  (v1.4.0+). They need regeneration; `scripts/coarsen_sigma2_budgets.py` now stamps
  `v1.4.0`, but `scripts/coarsen_sigma2_tracers.py` may still read an older string.
- The existing per-worktree conda environments predate the `CM4Xutils_<name>` naming
  convention and do not follow it, though each is correctly editable-installed against
  its own worktree: `cm4xutils-xeos` → `xwmt-xeos-eos`, `CM4Xutils-rho2-transports` →
  `rho2-native-transports`, `CM4Xutils-review` → `readability-review`. The base
  `CM4Xutils` env is installed against the main checkout. Rename them when convenient.
- This `CLAUDE.md` is untracked, so it exists only in the main checkout — worktrees do
  not get a copy, and sessions started inside one will not load these instructions.
  Track it, or copy it in, if that becomes a problem.
- `CM4Xutils/.ipynb_checkpoints/` is untracked and gitignored. As of v1.3.0 its module
  copies are absorbed into the live modules, **except** `new_loading-checkpoint.py`, which
  is the only surviving copy of a removed `new_loading` module
  (`load_averages_and_snapshots`, `swap_rho2_for_sigma2`, `load_CM4X_diags`). Don't delete
  that one without checking; exclude the directory when grepping.
