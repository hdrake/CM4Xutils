#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
from remap_functions import remap_tracers_to_sigma2_and_coarsen

# model options: ["CM4Xp25", "CM4Xp125"]
model = sys.argv[1]
# experiment options: ["piControl-spinup", "piControl", "historical", "ssp585"]
experiment = sys.argv[2]
# interval_start options: multiples of 5, starting with 101 (control) and 1850 (forced)
interval_start = np.int64(sys.argv[3])
# interval_length options: multiples of 5
interval_length = np.int64(sys.argv[4])

for start_year in np.arange(interval_start, interval_start+interval_length, 5):
    year_range = f"{str(start_year).zfill(4)}-{str(start_year+4).zfill(4)}"
    print(f"Processing tracers for {year_range}", end="\n")

    filename = f"../data/coarsened/{model}_{experiment}_tracers_sigma2_{year_range}.zarr"
    ds = remap_tracers_to_sigma2_and_coarsen(model, experiment, start_year)
    ds.attrs["version"] = "v0.2.0"
    ds.attrs["version_notes"] = """Between v0.1.0 and v0.2.0, CM4Xutils has been upgraded to v0.8.0. This includes two major bug fixes that change answers: 1) the coarsening bug has been fixed and 2) the target density grid has been extended so that extremely dense water masses are now represented in the target grid and mass is thus now properly conserved."""
    ds.to_zarr(filename, mode="w")
