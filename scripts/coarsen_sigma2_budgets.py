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
    ds.attrs["version"] = "v1.4.0"
    ds.attrs["version_notes"] = """In v1.4.0, the mass transports ("umo" and "vmo") are taken directly from the model's native density-coordinate ("ocean_month_rho2") diagnostics where available, instead of being remapped offline from z-coordinates. MOM6 accumulates these transports online into potential-density layers, conserving mass exactly within each layer, so they are much more accurate than the previous (<=v1.3.0) offline z->sigma2 remapping. This applies to CM4Xp125, which archives both "umo" and "vmo" in "ocean_month_rho2". CM4Xp25 archives only "vmo" (no "umo") in density coordinates, so its "umo"/"vmo" continue to use the offline remap and are unchanged from v1.3.0. Only the CM4Xp125 "umo"/"vmo" fields change; all other diagnostics are unaffected."""
    ds.to_zarr(filename, mode="w")
