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
    ds.attrs["version"] = "v0.4.0"
    ds.attrs["version_notes"] = """Between v0.3.0 and v0.4.0, CM4Xutils has been upgraded to v0.7.0. The main change is that the 'd2' coarsening bug (see https://github.com/hdrake/CM4Xutils/issues/6) has been fixed. This was a bug in the online MOM6 diagnostic manager, which used incorrect masking of weights in the area-weighted averaging of diagnostics with the 'xh:mean yh:mean z_l:sum' cell methods. This affected may of the foundational variables for water mass budget analysis, including the thickness of coarsened cells 'thkcello' and all heat and salt budget tendencies. This bug only affects the CM4Xp125 diagnostics, because they were downsampled by a factor of 2x2 (i.e. 'd2') whereas CM4Xp25 was output at its full native resolution."""
    ds.to_zarr(filename, mode="w")