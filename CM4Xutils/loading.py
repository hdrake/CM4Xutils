import numpy as np
import dask
import xarray as xr
import xwmt
import xgcm
import doralite
import gfdl_utils.core as gu

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
        "piControl-spinup" : "odiv-209",
        "piControl"  : "odiv-313",
        "historical" : "odiv-255",
        "ssp585"     : "odiv-293",
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

def load_wmt_averages_and_snapshots(model, exp, time="*", dmget=False, mirror=False):    
    pdict_tend = get_wmt_pathDict(model, exp, "tendency", time=time)
    pdict_surf = get_wmt_pathDict(model, exp, "surface" , time=time, add=["tos", "sos"])
    av_tend = gu.open_frompp(**pdict_tend, dmget=dmget, mirror=mirror)
    av_surf = gu.open_frompp(**pdict_surf, dmget=dmget, mirror=mirror)
    averages = xr.merge([av_tend, av_surf]).chunk({"time":1, "z_l":-1, "yh":-1, "xh":-1, "yq":-1, "xq":-1})

    pdict_snap = get_wmt_pathDict(model, exp, "snapshot", time=time)
    snapshots = gu.open_frompp(**pdict_snap, dmget=dmget, mirror=mirror).chunk({"time":1, "z_l":-1, "yh":-1, "xh":-1})
    snapshots = snapshots.rename({
        **{'time':'time_bounds'},
        **{v:f"{v}_bounds" for v in snapshots.data_vars}
    })
    
    return xr.merge([averages, snapshots])

def load_wmt_diags_CM4X(model, test=False, dmget=False, mirror=False, interval="all"):
    if test:
        time =      "185001*"
        time_ctrl = "010101*"
        interval  = "1850"
        load_hist = True
        load_ssp5 = False
    elif interval=="all":
        time = "*"
        time_ctrl = "*"
        load_hist = True
        load_ssp5 = True
    elif interval.isnumeric():
        if (int(interval)%5)==0:
            time = f"{interval}01*"
            time_ctrl = f"{str(int(interval)-1749).zfill(4)}01*"
            load_hist = int(interval) < 2015
            load_ssp5 = int(interval) >= 2015
        else:
            raise ValueError("interval must be an integer multiple of 5.")
            
    # Load mass/heat/salt budget diagnostics align times
    print(f"Loading {model}-piControl.")
    ctrl = load_wmt_averages_and_snapshots(
        model,
        "piControl",
        time=time_ctrl,
        dmget=dmget,
        mirror=mirror
    )
    
    if load_hist:
        print(f"Loading {model}-historical.")
        hist = load_wmt_averages_and_snapshots(
            model,
            "historical",
            time=time,
            dmget=dmget,
            mirror=mirror
        )
    if load_ssp5:
        print(f"Loading {model}-ssp585.")
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

    ctrl = align_dates(ctrl, forc, interval=interval)
    ds = xr.concat([
        forc.expand_dims({'exp': ["forced"]}),
        ctrl.expand_dims({'exp': ["control"]})
    ], dim="exp", combine_attrs="override")
    
    if test:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # Only keep second full year, to keep data load light
            ds = ds.isel(time=slice(12,24), time_bounds=slice(11, 24))

    print(f"Assigning {model} grid coordinates.")
    path_dict = get_wmt_pathDict(model, "piControl", "surface")
    og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], path_dict["ppname"]))
    sg = xr.open_dataset(exp_dict[model]["hgrid"])
    
    og = fix_geo_coords(og, sg)
    ds = add_grid_coords(ds, og)
    grid = ds_to_grid(ds)

    # Compute potential density variables
    if model=="CM4Xp125":
        coords = {'Z': {'center': 'z_l', 'outer': 'z_i'}}
        wm_kwargs = {"coords": coords, "metrics":{}, "boundary":{"Z":"extend"}, "autoparse_metadata":False}
        wm_averages = xwmt.WaterMass(xgcm.Grid(grid._ds[["thetao", "so", "thkcello", "z_i"]], **wm_kwargs))
        grid._ds["sigma2"] = wm_averages.get_density("sigma2")
        snapshot_state_vars = grid._ds[["thetao_bounds", "so_bounds", "thkcello_bounds", "z_i"]]
        wm_snapshots = xwmt.WaterMass(xgcm.Grid(snapshot_state_vars.rename({v:v.split("_")[0] for v in snapshot_state_vars.data_vars}), **wm_kwargs))
        grid._ds["sigma2_bounds"] = wm_snapshots.get_density("sigma2")
    else:
        grid._ds = swap_rho2_for_sigma2(grid._ds)
    
    return grid

def concat_scenarios(ds_list):
    return xr.merge([
        xr.concat([
            ds.drop_dims([dim for dim in ds.dims if dim!=cdim])
            for ds in ds_list
        ], dim=cdim, combine_attrs="override")
        for cdim in ds_list[0].dims if "time" in cdim
    ], combine_attrs="override")

def align_dates(ctrl, forc, interval="all"):
    # Align dates of a control simulation (with nominal dates starting from year 0)
    # to a historically-referenced simulation (e.g. with dates starting from 1850)
    if interval=="all":
        # Control is longer than forced experiments for some reason
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            ctrl = ctrl.sel(
                time=ctrl.time[:-120],
                time_bounds=ctrl.time_bounds[:-120]
            )

    ctrl_times, ctrl_time_bounds = ctrl.time.values.copy(), ctrl.time_bounds.values.copy()
    ctrl = ctrl.assign_coords({
        'time': xr.DataArray(forc.time.values, dims=("time",)),
        'time_bounds': xr.DataArray(forc.time_bounds.values, dims=("time_bounds",)),
    })
    # Keep record of original control dates for reference
    ctrl = ctrl.assign_coords({
        'time_original': xr.DataArray(ctrl_times, dims=("time",)),
        'time_bounds_original': xr.DataArray(ctrl_time_bounds, dims=("time_bounds",)),
    })

    return ctrl