import numpy as np
import dask
import xarray as xr
import xwmt
import xgcm
import doralite
import gfdl_utils.core as gu
import cftime

from .grid_preprocess import *

exp_dict = {
    "CM4Xp25": {
        "hgrid": (
            "/archive/Raphael.Dussin/datasets/OM4p25/"
            "c192_OM4_025_grid_No_mg_drag_v20160808_unpacked/"
            "ocean_hgrid.nc"
        ),
        "piControl-spinup": "odiv-210",
        "piControl"       : "odiv-230",
        "historical"      : "odiv-231",
        "ssp585"          : "odiv-232"
    },
    "CM4Xp125": {
        "hgrid": (
            "/archive/Raphael.Dussin/datasets/OM4p125"
            "/mosaic_c192_om4p125_bedmachine_v20210310_hydrographyKDunne20210614_unpacked/"
            "ocean_hgrid.nc"
        ),
        "piControl-spinup": "odiv-209",
        "piControl"       : "odiv-313",
        "historical"      : "odiv-255",
        "ssp585"          : "odiv-293",
    }
}

def get_wmt_pathDict(model, exp, category, time="*", add="*"):
    pp = doralite.dora_metadata(exp_dict[model][exp])['pathPP']
    freq = "month"
    ignore = ["1x1deg"]
    coarsen = ["d2"] if model=="CM4Xp125" else []
    if category=="surface":
        ppname = gu.find_unique_variable(pp, "tos", require=[freq]+coarsen, ignore=ignore)
    elif category=="tendency":
        ppname = gu.find_unique_variable(pp, "opottemptend", require=[freq]+coarsen, ignore=ignore)
    elif category=="snapshot":
        ppname = gu.find_unique_variable(pp, "thetao", require=[freq]+["snap"]+coarsen, ignore=ignore)
    else:
        raise ValueError("Valid categories are 'surface', 'tendency', and 'snapshot'.")
    local = gu.get_local(pp, ppname, "ts")
    return {
        "pp": pp,
        "ppname": ppname,
        "out": "ts",
        "local": local,
        "time": time,
        "add": add
    }

chunk = {"time":1, "z_l":-1, "yh":-1, "xh":-1, "yq":-1, "xq":-1}
chunk_center = {"time":1, "z_l":-1, "yh":-1, "xh":-1}
def load_wmt_averages_and_snapshots(model, exp, time="*", dmget=False, mirror=False):    
    pdict_tend = get_wmt_pathDict(model, exp, "tendency", time=time)
    pdict_surf = get_wmt_pathDict(model, exp, "surface" , time=time, add=["tos", "sos"])
    av_tend = gu.open_frompp(**pdict_tend, dmget=dmget, mirror=mirror)
    av_surf = gu.open_frompp(**pdict_surf, dmget=dmget, mirror=mirror)
    averages = xr.merge([av_tend, av_surf]).chunk(chunk)

    # Case 1: We are either reading in all times or just the first 5-year interval.
    # In either case, there is no prior 5 yr interval, so we're missing the initial snapshot.
    if (time=="*") | (time=="000101*"):
        pdict_snap = get_wmt_pathDict(model, exp, "snapshot", time=time)
        snapshots = gu.open_frompp(**pdict_snap, dmget=dmget, mirror=mirror).chunk(chunk_center)
    
    # Case 2: we are only reading in a specific 5 yr interval,
    # in which case we also need the last snapshot from the prior interval.
    elif (time!="*") & (time!="000101*"):
        interval_preceeding = str(np.int64(time.split("*")[0][:-2]) - 5).zfill(4)
        # Awkward corner cases where the prior interval is actually from a different experiment
        if interval_preceeding in ["0096", "1845"]:
            pdict_snap_preceeding = get_wmt_pathDict(
                model, "piControl-spinup", "snapshot", time=f"009601*"
            )
        elif interval_preceeding=="2010":
            pdict_snap_preceeding = get_wmt_pathDict(
                model, "historical", "snapshot", time=f"201001*"
            )
        else:
            pdict_snap_preceeding = get_wmt_pathDict(
                model, exp, "snapshot", time=f"{interval_preceeding}01*"
            )
        pdict_snap = get_wmt_pathDict(model, exp, "snapshot", time=time)
        snapshots = xr.concat(
            [
                gu.open_frompp(**pdict_snap_preceeding, dmget=dmget, mirror=mirror).chunk(chunk_center).isel(time=-1),# only the last
                gu.open_frompp(**pdict_snap, dmget=dmget, mirror=mirror).chunk(chunk_center)
            ],
            dim="time"
        )

    snapshots = snapshots.rename({
        **{'time':'time_bounds'},
        **{v:f"{v}_bounds" for v in snapshots.data_vars}
    })
    
    return xr.merge([averages, snapshots])

def load_wmt_grid(model, **kwargs):
    ds = load_wmt_ds(model, **kwargs)
    grid = make_grid(ds)
    return grid

def load_wmt_ds(model, test=False, dmget=False, mirror=False, interval="all"):
    if test:
        time =      "201001*"
        time_ctrl = "026101*"
        interval  = "2010"
        load_spin = False
        load_ctrl = True
        load_hist = True
        load_ssp5 = False
    elif interval=="all":
        time = "*"
        time_ctrl = "*"
        load_spin = True
        load_ctrl = True
        load_hist = True
        load_ssp5 = True
    elif interval.isnumeric():
        if (int(interval)%5)==0:
            time = f"{interval}01*"
            time_ctrl = f"{str(int(interval)-1749).zfill(4)}01*"
            load_spin = int(interval) < 1850
            load_ctrl = 1850 <= int(interval)
            load_hist = (1850 <= int(interval)) & (int(interval) < 2015)
            load_ssp5 = 2015 <= int(interval)
        else:
            raise ValueError("interval must be an integer multiple of 5.")
            
    # Load mass/heat/salt budget diagnostics align times
    if load_spin:
        print(f"Loading {model}-piControl-spinup for interval `{interval}`.")
        spinup = load_wmt_averages_and_snapshots(
            model,
            "piControl-spinup",
            time=time_ctrl,
            dmget=dmget,
            mirror=mirror
        )

    if load_ctrl:
        print(f"Loading {model}-piControl for interval `{interval}`.")
        ctrl = load_wmt_averages_and_snapshots(
            model,
            "piControl",
            time=time_ctrl,
            dmget=dmget,
            mirror=mirror
        )
    
    if load_hist:
        print(f"Loading {model}-historical for interval `{interval}`.")
        hist = load_wmt_averages_and_snapshots(
            model,
            "historical",
            time=time,
            dmget=dmget,
            mirror=mirror
        )

    if load_ssp5:
        print(f"Loading {model}-ssp585 for interval `{interval}`.")
        ssp5 = load_wmt_averages_and_snapshots(
            model,
            "ssp585",
            time=time,
            dmget=dmget,
            mirror=mirror
        )

    if load_hist and load_ssp5:
        forc = concat_scenarios(hist, ssp5)
    elif load_hist:
        forc = hist
    elif load_ssp5:
        forc = ssp5
    
    # The historical only branches off after the spin-up, we need to expand the
    # exp dimension of the spinup and specify it for both the control and forced.

    # Case 1: We are only loading in intervals from the spinup
    if load_spin and not(load_ctrl):
        ds = xr.concat([
            spinup.expand_dims({'exp': ["forced"]}),
            spinup.expand_dims({'exp': ["control"]})
        ], dim="exp", combine_attrs="override")

    # Case 2: We are loading in intervals from the control period (including forced runs)
    elif load_ctrl:
        ctrl, forc = align_dates(ctrl, forc)
        ds = xr.concat([
            forc.expand_dims({'exp': ["forced"]}),
            ctrl.expand_dims({'exp': ["control"]})
        ], dim="exp", combine_attrs="override")
        
        if load_spin:
            spinup = xr.concat([
                spinup.expand_dims({'exp': ["forced"]}),
                spinup.expand_dims({'exp': ["control"]})
            ], dim="exp", combine_attrs="override")
            ds = concat_scenarios(spinup, ds)
    
    if test:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # Only keep second full year, to keep data load light
            ds = ds.isel(time=slice(12,24), time_bounds=slice(11, 24))

    ds.coords["exp"].attrs = {"long_name": "Experiment type"}
    ds.attrs["model"] = model
    ds.attrs["description"] = (
    f"""The {model} experimental design following Griffies et al.
(to be submitted to JAMES around 09/2024).

The `control` experiment type is a ocean-sea ice-atmosphere-land coupled
climate model run with CO2 concentrations in the atmosphere
prescribed at 280 ppm (preindustrial levels).

The `forced` experiment type branches off from the preindustrial control
in 1850 and is forced with historical CMIP6 forcings until 2014 and afterwards
follows the SSP5-8.5 high-emissions forcing scenario."""
    )
    
    return ds

def make_grid(ds):
    print(f"Assigning {ds.attrs["model"]} grid coordinates.")
    path_dict = get_wmt_pathDict(ds.attrs["model"], "piControl", "surface")
    og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], path_dict["ppname"]))
    sg = xr.open_dataset(exp_dict[ds.attrs["model"]]["hgrid"])

    attrs = {c:ds.coords[c].attrs.copy() for c in ds.coords}
    
    og = fix_geo_coords(og, sg)
    ds = add_grid_coords(ds, og)
    grid = ds_to_grid(ds)

    # Compute potential density variables
    coords = {'Z': {'center': 'z_l', 'outer': 'z_i'}}
    wm_kwargs = {"coords": coords, "metrics":{}, "boundary":{"Z":"extend"}, "autoparse_metadata":False}
    wm_averages = xwmt.WaterMass(xgcm.Grid(grid._ds[["thetao", "so", "thkcello", "z_i"]], **wm_kwargs))
    grid._ds["sigma2"] = wm_averages.get_density("sigma2")
    snapshot_state_vars = grid._ds[["thetao_bounds", "so_bounds", "thkcello_bounds", "z_i"]]
    rename_vardict = {v:v.split("_")[0] for v in snapshot_state_vars.data_vars}
    wm_snapshots = xwmt.WaterMass(xgcm.Grid(snapshot_state_vars.rename(rename_vardict), **wm_kwargs))
    grid._ds["sigma2_bounds"] = wm_snapshots.get_density("sigma2")

    for (c,a) in attrs.items():
        grid._ds.coords[c].attrs = a
    
    return grid

def concat_scenarios(ds_list):
    return xr.merge([
        xr.concat([
            ds.drop_dims([dim for dim in ds.dims if (dim!=cdim) & ("time" in dim)])
            for ds in ds_list
        ], dim=cdim, combine_attrs="override")
        for cdim in ds_list[0].dims if "time" in cdim
    ], combine_attrs="override")

def align_dates(ds_ctrl, ds_hist):
    # Align dates of a control simulation (with nominal dates starting from year 0)
    # to a historically-referenced simulation (e.g. with dates starting from 1850)
    for c in ["time", "time_bounds"]:
        if not np.all([c in d for d in [ds_ctrl.dims, ds_hist.dims]]): continue
        time_ctrl = ds_ctrl[c].copy()

        # Need to address corner cases where spinup snapshots get added to piControl
        historical_equivalent_dates = np.array([
            cftime.DatetimeNoLeap(d.dt.year+1749,d.dt.month,d.dt.day,0,0,0,0,has_year_zero=True)
            for d in ds_hist[c]
        ])
        c_corrected = xr.where(
            ds_hist[c].dt.year < 1850,
            historical_equivalent_dates,
            ds_hist[c]
        )
        ds_hist = ds_hist.assign_coords({c: c_corrected})
            
        hist_years = ds_hist[c].dt.year.values
        ctrl_years = time_ctrl.dt.year.values
        ctrl_years = (ctrl_years + (hist_years[0] - ctrl_years[0]))
        ctrl_years_mask = np.array([y in hist_years for y in ctrl_years])
        hist_years_mask = np.array([y in ctrl_years for y in hist_years])
        ds_ctrl = ds_ctrl.isel({c:ctrl_years_mask})
        ds_hist = ds_hist.isel({c:hist_years_mask})
    
        ds_ctrl = ds_ctrl.assign_coords({
            c: ds_hist[c],
            f"{c}_since_init": time_ctrl.isel({c:ctrl_years_mask})
        })

    ds_hist.coords["time"].attrs = {
        **ds_ctrl.coords["time_since_init"].attrs,
        **{"long_name": "historical time", "cell_methods": "time:mean"}
    }
    ds_hist.coords["time_bounds"].attrs = {
        **ds_ctrl.coords["time_bounds_since_init"].attrs,
        **{"long_name": "historical time", "cell_methods": "time:point"}
    }

    ds_ctrl.coords["time_since_init"].attrs = {
        **ds_ctrl.coords["time_since_init"].attrs,
        **{"long_name": "time since model initialization", "cell_methods": "time:mean"}
    }
    ds_ctrl.coords["time_bounds_since_init"].attrs = {
        **ds_ctrl.coords["time_bounds_since_init"].attrs,
        **{"long_name": "time since model initialization", "cell_methods": "time:point"}
    }
        
    return ds_ctrl, ds_hist
