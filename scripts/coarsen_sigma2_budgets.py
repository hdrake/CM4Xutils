#!/usr/bin/env python
# coding: utf-8
import sys
import numpy as np
from remap_functions import remap_budgets_to_sigma2_and_coarsen

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
    ds.attrs["version"] = "v1.0.0"
    ds.attrs["version_notes"] = """Between v0.5.0 and v1.0.0, CM4Xutils has been upgraded to major release v1.0.0."""
    ds.to_zarr(filename, mode="w")
