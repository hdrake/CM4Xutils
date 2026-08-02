#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
from CM4Xutils import __version__
from remap_functions import remap_budgets_to_sigma2_and_coarsen

# The dataset release is versioned in lockstep with the package that generated it, so
# `version` is always derived from `CM4Xutils/version.py` rather than hard-coded here.
# When a change alters the numerical output, bump the package version and update
# `version_notes` (which describes what changed relative to the previous release).
version_notes = f"""As of v{__version__}, the offline potential density (sigma2) coordinate is computed with the MOM6 Wright (1997) reduced-range equation of state via xeos (xwmt eos="wright97-reduced"), self-consistently matching the CM4X model configuration EQN_OF_STATE="WRIGHT", instead of the gsw/TEOS-10 implementation used in earlier releases. This shifts sigma2 (and everything binned into it) by O(0.01-0.1 kg/m3). It matches the online model density to machine precision (~1e-12 kg/m3)."""

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
    ds = ds.chunk({"time":1, "time_bounds":1})
    ds.attrs["version"] = f"v{__version__}"
    ds.attrs["version_notes"] = version_notes
    ds.to_zarr(filename, mode="w")
