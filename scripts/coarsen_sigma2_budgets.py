#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
from CM4Xutils import __version__, chunk_dataset, finalize_metadata
from remap_functions import remap_budgets_to_sigma2_and_coarsen

# The dataset release is versioned in lockstep with the package that generated it, so
# `version` is always derived from `CM4Xutils/version.py` rather than hard-coded here.
# When a change alters the numerical output, bump the package version and update
# `version_notes` (which describes what changed relative to the previous release).
version_notes = f"""v{__version__} makes two changes relative to v1.3.0. (1) The offline potential density (sigma2) coordinate is computed with the MOM6 Wright (1997) reduced-range equation of state via xeos (xwmt eos="wright97-reduced"), self-consistently matching the CM4X model configuration EQN_OF_STATE="WRIGHT", instead of the gsw/TEOS-10 implementation used in earlier releases. This shifts sigma2 (and everything binned into it) by O(0.01-0.1 kg/m3). It matches the online model density to machine precision (~1e-12 kg/m3). (2) The mass transports ("umo" and "vmo") are taken directly from the model's online-remapped density-coordinate ("ocean_month_rho2") diagnostics where available, instead of being remapped offline from z-coordinates. Density is not the model's native vertical coordinate: MOM6 remaps these transports into potential-density layers online, at every timestep and using the instantaneous density, conserving mass exactly within each layer, so they are much more accurate than the previous (<=v1.3.0) offline remapping of time-mean z-coordinate transports. This applies to CM4Xp125, which archives both "umo" and "vmo" in "ocean_month_rho2". CM4Xp25 archives only "vmo" (no "umo") in density coordinates, so its "umo"/"vmo" continue to use the offline remap -- though they, like every other density-binned field, still shift because of change (1)."""

# model options: ["CM4Xp25", "CM4Xp125"]
model = sys.argv[1]
# interval_start options: multiples of 5 between 1750 and 2095
interval_start = np.int64(sys.argv[2])
# interval_length options: multiples of 5
interval_length = np.int64(sys.argv[3])

for start_year in np.arange(interval_start, interval_start+interval_length, 5):
    year_range = f"{str(start_year).zfill(4)}-{str(start_year+4).zfill(4)}"
    print(f"Processing budgets for {year_range}", end="\n")
    
    filename = f"../data/coarsened/{model}_budgets_sigma2_{year_range}.zarr"
    ds = remap_budgets_to_sigma2_and_coarsen(model, start_year)
    ds = chunk_dataset(ds, {"time":1, "time_bounds":1})
    # `finalize_metadata` owns every descriptive attribute, including the dataset
    # release version -- it writes a machine-readable `product_version` (bare semver)
    # and `source_software`, so the version is no longer restated as a separate
    # hand-formatted `version` string that could drift from it.
    ds = finalize_metadata(
        ds,
        model=model,
        product="budgets",
        time_range=year_range,
        version_notes=version_notes,
    )
    ds.to_zarr(filename, mode="w")
