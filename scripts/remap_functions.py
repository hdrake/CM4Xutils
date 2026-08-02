
import warnings

import numpy
import dask
import xarray
import matplotlib.pyplot as plt

import doralite
import gfdl_utils.core as gu
from CM4Xutils import *

def remap_budgets_to_sigma2_and_coarsen(model, start_year):

    coarsen_dims = {
        "CM4Xp25": {"X": 6, "Y": 6},
        "CM4Xp125": {"X": 6, "Y": 5}, # Note: these are already on d2 grid
    }
    # Native rho2 transports are archived at native resolution, so for CM4Xp125 they need
    # twice the coarsening of the d2 budget diagnostics to land on the same coarse grid
    # (d2 == native coarsened by 2).
    transport_coarsen_dims = {
        "CM4Xp25": {"X": 6, "Y": 6},
        "CM4Xp125": {"X": 12, "Y": 10},
    }

    # Only source transports from the native rho2 diagnostics when BOTH `umo` and `vmo`
    # are archived across all experiments in this interval. CM4Xp125 saves both; CM4Xp25
    # saves only `vmo`, so CM4Xp25 keeps the offline z->sigma2 transport remapping for
    # both terms (auto-detected, not hard-coded).
    native_vars = native_rho2_transport_vars(model, interval=str(start_year))
    use_native_transports = native_vars == {"umo", "vmo"}
    if use_native_transports:
        print(f"Using native rho2 transports (umo, vmo) for {model}.")
    else:
        print(
            f"Native rho2 transports unavailable for {model} "
            f"(found {sorted(native_vars)}); using offline z->sigma2 remap for umo/vmo."
        )

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)

        grid = load_wmt_grid(
            model,
            interval=str(start_year),
            dmget=True
        )

        ds = add_sigma2_coords(grid._ds)
        vars_2d = [v for v in ds.data_vars if sorted(ds[v].dims) == ['exp', 'time', 'xh', 'yh']]
        # When using native transports, skip the offline z->sigma2 remap of umo/vmo.
        ds_sigma2 = xr.merge([
            remap_vertical_coord(
                "sigma2", ds, grid,
                remap_transports=not use_native_transports
            ),
            ds[vars_2d]
        ])
        grid_sigma2 = ds_to_grid(ds_sigma2)

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds_sigma2_coarse = horizontally_coarsen(
            ds_sigma2,
            grid_sigma2,
            dim = coarsen_dims[model]
        )
        ds_sigma2_coarse = ds_sigma2_coarse.assign_coords(
            {"sigma2_i": ds_sigma2.coords["sigma2_i"]}
        )

    # --- Native density-coordinate transports (umo, vmo) from ocean_month_rho2 ---
    if use_native_transports:
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            warnings.simplefilter(action='ignore', category=UserWarning)

            ds_trans = load_rho2_transports_ds(
                model,
                interval=str(start_year),
                dmget=True
            )
            ds_trans = rho2_transports_to_sigma2(
                ds_trans,
                ds_sigma2.coords["sigma2_l"],
                ds_sigma2.coords["sigma2_i"],
            )
            grid_trans = ds_to_grid(ds_trans)

        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            ds_trans_coarse = horizontally_coarsen(
                ds_trans,
                grid_trans,
                dim = transport_coarsen_dims[model]
            )

        # Adopt the budget product's horizontal coordinates (identical geometry up to
        # floating point) so umo/vmo slot cleanly into the same coarse grid.
        shared_coords = [
            "xh", "yh", "xq", "yq",
            "geolon", "geolat", "geolon_u", "geolat_u",
            "geolon_v", "geolat_v", "geolon_c", "geolat_c",
            "areacello", "wet", "wet_u", "wet_v", "dxCv", "dyCu",
        ]
        ds_trans_coarse = ds_trans_coarse.assign_coords({
            c: ds_sigma2_coarse.coords[c]
            for c in shared_coords
            if (c in ds_sigma2_coarse.coords) and (c in ds_trans_coarse.coords)
        })

        # Insert onto the budget product's dims/coords without triggering label alignment
        # (both went through the same `align_dates`, so positions already correspond).
        umo = ds_trans_coarse["umo"].transpose("exp", "time", "sigma2_l", "yh", "xq")
        vmo = ds_trans_coarse["vmo"].transpose("exp", "time", "sigma2_l", "yq", "xh")
        ds_sigma2_coarse["umo"] = xr.DataArray(umo.data, dims=umo.dims, attrs=umo.attrs)
        ds_sigma2_coarse["vmo"] = xr.DataArray(vmo.data, dims=vmo.dims, attrs=vmo.attrs)

    ordered_dims = [
        'exp', 'time', 'time_bounds', 'sigma2_l', 'sigma2_i',
        'yh', 'yq', 'xh', 'xq'
    ]
    ds_sigma2_coarse = ds_sigma2_coarse.transpose(*ordered_dims)

    return chunk_dataset(
        ds_sigma2_coarse,
        {d:-1 for d in ds_sigma2_coarse.dims if "time" not in d}
    )


def remap_tracers_to_sigma2_and_coarsen(model, experiment, start_year):
    odiv = exp_dict[model][experiment]
    time = str(start_year).zfill(4)+"*"
    
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
    
    # Load ideal age (for CM4Xp125, it is only available at 2x2 coarsened grid)
    age = load_tracer(odiv, "agessc", time=time)
    
    # Interpolate from annual-means to monthly-means (to match other transient tracers)
    interp_kwargs = {"fill_value": "extrapolate"}
    age = chunk_dataset(
        age.interp(time=ds.time, kwargs=interp_kwargs),
        {"time":1, "z_l":-1}
    )
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
    grid_age = ds_to_grid(ds_age, Zprefix="z_")
    
    # Transform to density coordinates
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
    
        ds_age_sigma2 = remap_vertical_coord("sigma2", chunk_dataset(ds_age, {"z_l":-1}), grid_age)
        grid_age_sigma2 = ds_to_grid(ds_age_sigma2)

    # Coarsen
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        ds_sigma2_coarse = horizontally_coarsen(
            ds_sigma2,
            grid_sigma2,
            coarsen_dims[model]
        )
        ds_sigma2_coarse = ds_sigma2_coarse.assign_coords({"sigma2_i": ds_sigma2.coords["sigma2_i"]})
        
        dim = {
            k:v//2 if model=="CM4Xp125" else v
            for (k,v) in coarsen_dims[model].items()
        }
        ds_age_sigma2_coarse = horizontally_coarsen(
            ds_age_sigma2,
            grid_age_sigma2,
            dim
        )
    
        ds_age_sigma2_coarse = (
            ds_age_sigma2_coarse.assign_coords(ds_sigma2_coarse.coords)
        )
        ds_sigma2_coarse["agessc"] = ds_age_sigma2_coarse["agessc"]

    ordered_dims = ['time', 'sigma2_l', 'sigma2_i', 'yh', 'yq', 'xh', 'xq']
    ds_sigma2_coarse = ds_sigma2_coarse.transpose(*ordered_dims)

    return chunk_dataset(
        ds_sigma2_coarse,
        {d:-1 for d in ds_sigma2_coarse.dims if "time" not in d}
    )
