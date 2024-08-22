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
# experiment options: ["piControl-spinup", "piControl", "historical", "ssp585"]
experiment = sys.argv[2]

def remap_tracers_to_sigma2_and_coarsen(model, experiment, time="*"):
    odiv = exp_dict[model][experiment]

    # Coarsening factors
    coarsen_dims = {
        "CM4Xp25": {"X": 6, "Y": 6},
        "CM4Xp125": {"X": 12, "Y": 10},
    }
    
    # Load transient tracers and state 
    grid = load_transient_tracers(odiv, time=time)
    ds = grid._ds
    ds = add_sigma2_coords(ds)
    
    # Transform transient to density coordinates
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        ds_sigma2 = remap_vertical_coord("sigma2", ds, grid)
        grid_sigma2 = ds_to_grid(ds_sigma2)
    
    # Load ideal age (for CM4Xp125, only available at 2x2 coarsened grid)
    age = load_tracer(odiv, "agessc", time=time)
    
    # Interpolate from annual-means to monthly-means (to match other transient tracers)
    interp_kwargs = {"fill_value": "extrapolate"}
    age = age.interp(time=ds.time, kwargs=interp_kwargs).chunk({"time":1})
    age["average_DT"] = ds.average_DT
    
    # Zero-out negative ages (due to extrapolation of time interpolation in first few months)
    age["agessc"] = age.agessc.where(
        (age.agessc >= 0) | np.isnan(age.agessc),
        0.
    )
    
    if model == "CM4Xp125":
        # Coarsen layer thickness and density to match "d2" resolution of ideal age output 
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds_age = ds.drop_vars(
                [v for v in ds.data_vars if v not in ["sigma2", "thkcello"]]
            )
            ds_age = horizontally_coarsen(ds_age, grid, {"X":2, "Y":2})
            ds_age = ds_age.assign_coords({
                k:v for (k,v) in ds.coords.items()
                if k in ["sigma2_l", "sigma2_i", "z_i"]
            })
    else:
        ds_age = ds
        
    # Overwrite coordinates with corrected coordinates in main dataset for smooth merging
    age = age.assign_coords(ds_age.coords)
    
    # Add age to other variables--need the coordinates of `ds_age.agessc` to match with those of `ds_d2`!
    ds_age["agessc"] = age.agessc
    grid_age = ds_to_grid(ds_age, Zprefix="z")
    
    # Transform to density coordinates
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
    
        ds_age_sigma2 = remap_vertical_coord("sigma2", ds_age, grid_age)
        grid_age_sigma2 = ds_to_grid(ds_age_sigma2)

    # Coarsen
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds_sigma2_coarse = horizontally_coarsen(
            ds_sigma2,
            grid_sigma2,
            coarsen_dims[model]
        )
        
        dim = {
            k:v//2 if model=="CM4Xp125" else v
            for (k,v) in coarsen_dims[model].items()
        }
        ds_age_sigma2_coarse = horizontally_coarsen(
            ds_age_sigma2,
            grid_age_sigma2,
            dim
        )
    
        ds_age_sigma2_coarse = ds_age_sigma2_coarse.assign_coords(ds_sigma2_coarse.coords)
        ds_sigma2_coarse["agessc"] = ds_age_sigma2_coarse["agessc"]

    return ds_sigma2_coarse.chunk({"xh":-1, "yh":-1, "sigma2_l":-1})

