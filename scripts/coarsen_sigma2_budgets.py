#!/usr/bin/env python
# coding: utf-8

import warnings

import numpy
import dask
import xarray
import matplotlib.pyplot as plt

import doralite
import gfdl_utils.core as gu
from CM4Xutils import *

import sys

# model options: ["CM4Xp25", "CM4Xp125"]
model = sys.argv[1]
# interval_start options: multiples of 5 between 1750 and 2095
interval_start = np.int64(sys.argv[2])
# interval_length options: multiples of 5
interval_length = np.int64(sys.argv[3])

coarsen_dims = {
    "CM4Xp25": {"X": 12, "Y": 12},
    "CM4Xp125": {"X": 12, "Y": 10},
}

def remap_budgets_to_sigma2_and_coarsen(model, interval_start, interval_length):
    for start_year in np.arange(interval_start, interval_start+interval_length, 5):
    
        year_range = f"{str(start_year)}-{str(start_year+4)}"
        print(f"Processing budgets for {year_range}", end="\n")
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            warnings.simplefilter(action='ignore', category=UserWarning)
        
            grid = load_wmt_grid(
                model,
                interval=str(start_year),
                dmget=True
            )
            ds = add_sigma2_coords(grid._ds)
            ds_sigma2 = xr.merge([
                remap_vertical_coord("sigma2", ds, grid),
                ds[["tos", "sos"]]
            ])
            grid_sigma2 = ds_to_grid(ds_sigma2)
    
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds_sigma2_coarse = horizontally_coarsen(
                ds_sigma2,
                grid_sigma2,
                dim = coarsen_dims[model]
            )
    
        ordered_dims = ['exp', 'time', 'time_bounds', 'sigma2_l', 'yh', 'yq', 'xh', 'xq']
        ds_sigma2_coarse = ds_sigma2_coarse.transpose(*ordered_dims)

        filename = f"../data/coarsened/{model}_budgets_sigma2_{year_range}.zarr"
        ds_sigma2_coarse.to_zarr(filename, mode="w")
        
remap_budgets_to_sigma2_and_coarsen(model, interval_start, interval_length)