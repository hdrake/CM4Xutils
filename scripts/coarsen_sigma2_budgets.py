#!/usr/bin/env python
# coding: utf-8
import os
import sys
import numpy as np
from CM4Xutils import (
    __version__, chunk_dataset, finalize_metadata, prune_cell_measures
)
from remap_functions import remap_budgets_to_sigma2_and_coarsen

# The dataset release is versioned in lockstep with the package that generated it:
# `finalize_metadata` writes `product_version` from `CM4Xutils/version.py`, so it can
# never be hard-coded here or drift from the code that produced the file. When a change
# alters the numerical output, bump the package version and update `version_notes`
# below (which describes what changed relative to the previous release).
version_notes = f"""v{__version__} makes two changes relative to v1.3.0. (1) The offline potential density (sigma2) coordinate is computed with the MOM6 Wright (1997) reduced-range equation of state via xeos (xwmt eos="wright97-reduced"), self-consistently matching the CM4X model configuration EQN_OF_STATE="WRIGHT", instead of the gsw/TEOS-10 implementation used in earlier releases. This shifts sigma2 (and everything binned into it) by O(0.01-0.1 kg/m3). It matches the online model density to machine precision (~1e-12 kg/m3). (2) The mass transports ("umo" and "vmo") are taken directly from the model's online-remapped density-coordinate ("ocean_month_rho2") diagnostics where available, instead of being remapped offline from z-coordinates. Density is not the model's native vertical coordinate: MOM6 remaps these transports into potential-density layers online, at every timestep and using the instantaneous density, conserving mass exactly within each layer, so they are much more accurate than the previous (<=v1.3.0) offline remapping of time-mean z-coordinate transports. This applies to CM4Xp125, which archives both "umo" and "vmo" in "ocean_month_rho2". CM4Xp25 archives only "vmo" (no "umo") in density coordinates, so its "umo"/"vmo" continue to use the offline remap -- though they, like every other density-binned field, still shift because of change (1)."""

# Months per `to_zarr` call. Peak memory grows steeply with the months written at once --
# measured one write per job, peak read from `sacct MaxRSS` (an in-process sampler thread
# is GIL-starved while numpy computes and undercounts by ~2x):
#
#     months     MaxRSS     s/month
#          1    27.1 GB         174
#          2    40.6 GB         194
#          4    46.6 GB         207
#          6    86.6 GB         195
#
# A single write of all 60 months does not fit on a 125 GB node. Throughput is flat at
# ~200 s/month regardless of batch size, so a smaller batch costs almost nothing: 4 months
# is the largest with real headroom, and a full interval is ~4 h.
TIME_BATCH = int(os.environ.get("CM4X_TIME_BATCH", "4"))


def write_in_time_batches(ds, path, months=TIME_BATCH):
    """Write `ds` a few months at a time, so peak memory tracks `months`, not the interval.

    Time is the right axis to batch along: the expensive shared intermediates are indexed
    by (exp, time), so a batch computes only its own months and nothing is duplicated.
    Batching by variable would instead recompute them for every subset.

    Two details are not optional:

    1. Coordinates are materialized first. Some are dask-backed, and
       `to_zarr(compute=False)` defers dask-backed variables -- coords included -- leaving
       them at their fill value; reopening the store to fill a region then tries to decode
       NaN as a date and raises `Failed to decode variable 'time_bounds_since_init'`.
    2. Region writes carry data variables only, since the skeleton already wrote the
       coords.
    """
    ds = ds.assign_coords({
        c: ds[c].compute() for c in ds.coords if hasattr(ds[c].data, "dask")
    })

    time_vars = [v for v in ds.data_vars if "time" in ds[v].dims]
    tb_vars = [v for v in ds.data_vars if "time_bounds" in ds[v].dims]
    other = [v for v in ds.data_vars if v not in time_vars and v not in tb_vars]

    ds.to_zarr(path, mode="w", compute=False)
    if other:
        ds[other].to_zarr(path, mode="a")

    for dim, varlist in (("time", time_vars), ("time_bounds", tb_vars)):
        if not varlist or dim not in ds.sizes:
            continue
        n = ds.sizes[dim]
        for i0 in range(0, n, months):
            i1 = min(i0 + months, n)
            sub = ds[varlist].isel({dim: slice(i0, i1)})
            sub = sub.drop_vars(list(sub.coords))
            sub.to_zarr(path, region={dim: slice(i0, i1)})
            print(f"    wrote {dim}[{i0}:{i1}] of {n}", flush=True)


def main():
    # model options: ["CM4Xp25", "CM4Xp125"]
    model = sys.argv[1]
    # interval_start options: multiples of 5 between 1750 and 2095
    interval_start = np.int64(sys.argv[2])
    # interval_length options: multiples of 5
    interval_length = np.int64(sys.argv[3])
    # Optional output directory, so a full release can be regenerated into its own
    # versioned folder (e.g. `../data/coarsened_v2.0.0`) without overwriting the
    # current one. Defaults to the historical location.
    outdir = sys.argv[4] if len(sys.argv) > 4 else "../data/coarsened"
    os.makedirs(outdir, exist_ok=True)

    for start_year in np.arange(interval_start,
                                interval_start + interval_length, 5):
        year_range = f"{str(start_year).zfill(4)}-{str(start_year+4).zfill(4)}"
        print(f"Processing budgets for {year_range}", end="\n")

        filename = os.path.join(
            outdir, f"{model}_budgets_sigma2_{year_range}.zarr"
        )
        ds = remap_budgets_to_sigma2_and_coarsen(model, start_year)
        ds = chunk_dataset(ds, {"time": 1, "time_bounds": 1})
        # `volcello` is dropped by `add_sigma2_coords`, so the `cell_measures`
        # inherited from the archived diagnostics would name a variable that is
        # not in the store.
        ds = prune_cell_measures(ds)
        # `finalize_metadata` owns every descriptive attribute, including the
        # dataset release version -- it writes a machine-readable
        # `product_version` (bare semver) and `source_software`, so the version
        # is no longer restated as a separate hand-formatted `version` string
        # that could drift from it.
        ds = finalize_metadata(
            ds,
            model=model,
            product="budgets",
            time_range=year_range,
            version_notes=version_notes,
        )
        write_in_time_batches(ds, filename, months=TIME_BATCH)


if __name__ == "__main__":
    main()
