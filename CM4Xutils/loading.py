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

def load_wmt_averages_and_snapshots(model, exp, time="*", dmget=False, mirror=False):    
    pdict_tend = get_wmt_pathDict(model, exp, "tendency", time=time)
    pdict_surf = get_wmt_pathDict(model, exp, "surface" , time=time, add=["tos", "sos"])
    av_tend = gu.open_frompp(**pdict_tend, dmget=dmget, mirror=mirror)
    av_surf = gu.open_frompp(**pdict_surf, dmget=dmget, mirror=mirror)
    averages = xr.merge([av_tend, av_surf]).chunk({"time":1, "z_l":-1, "yh":-1, "xh":-1, "yq":-1, "xq":-1})

    if (time!="*") & (time!="185001*") & (time!="010101*"):
        interval_preceeding = str(np.int64(time.split("*")[0][:-2]) - 5).zfill(4)
        if interval_preceeding=="2010": # Only awkward case!
            pdict_snap_preceeding = get_wmt_pathDict(model, "historical", "snapshot", time=f"{interval_preceeding}01*")
        else:
            pdict_snap_preceeding = get_wmt_pathDict(model, exp, "snapshot", time=f"{interval_preceeding}01*")
        pdict_snap = get_wmt_pathDict(model, exp, "snapshot", time=time)
        snapshots = xr.concat(
            [
                gu.open_frompp(**pdict_snap_preceeding, dmget=dmget, mirror=mirror).chunk({"time":1, "z_l":-1, "yh":-1, "xh":-1}).isel(time=-1),
                gu.open_frompp(**pdict_snap, dmget=dmget, mirror=mirror).chunk({"time":1, "z_l":-1, "yh":-1, "xh":-1})
            ],
            dim="time"
        )
    else:
        pdict_snap = get_wmt_pathDict(model, exp, "snapshot", time=time)
        snapshots = gu.open_frompp(**pdict_snap, dmget=dmget, mirror=mirror).chunk({"time":1, "z_l":-1, "yh":-1, "xh":-1})
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

    ctrl, forc = align_dates(ctrl, forc)
    ds = xr.concat([
        forc.expand_dims({'exp': ["forced"]}),
        ctrl.expand_dims({'exp': ["control"]})
    ], dim="exp", combine_attrs="override")
    
    if test:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # Only keep second full year, to keep data load light
            ds = ds.isel(time=slice(12,24), time_bounds=slice(11, 24))

    ds.attrs["model"] = model
    
    return ds

def make_grid(ds):
    print(f"Assigning {ds.attrs["model"]} grid coordinates.")
    path_dict = get_wmt_pathDict(ds.attrs["model"], "piControl", "surface")
    og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], path_dict["ppname"]))
    sg = xr.open_dataset(exp_dict[ds.attrs["model"]]["hgrid"])
    
    og = fix_geo_coords(og, sg)
    ds = add_grid_coords(ds, og)
    grid = ds_to_grid(ds)

    # Compute potential density variables
    coords = {'Z': {'center': 'z_l', 'outer': 'z_i'}}
    wm_kwargs = {"coords": coords, "metrics":{}, "boundary":{"Z":"extend"}, "autoparse_metadata":False}
    wm_averages = xwmt.WaterMass(xgcm.Grid(grid._ds[["thetao", "so", "thkcello", "z_i"]], **wm_kwargs))
    grid._ds["sigma2"] = wm_averages.get_density("sigma2")
    snapshot_state_vars = grid._ds[["thetao_bounds", "so_bounds", "thkcello_bounds", "z_i"]]
    wm_snapshots = xwmt.WaterMass(xgcm.Grid(snapshot_state_vars.rename({v:v.split("_")[0] for v in snapshot_state_vars.data_vars}), **wm_kwargs))
    grid._ds["sigma2_bounds"] = wm_snapshots.get_density("sigma2")

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
        ctrl_years = (time_ctrl.dt.year + (ds_hist[c].dt.year[0] - time_ctrl[c].dt.year[0])).values
        hist_years = ds_hist[c].dt.year.values
        ctrl_years_mask = np.array([y in hist_years for y in ctrl_years])
        hist_years_mask = np.array([y in ctrl_years for y in hist_years])
        ds_ctrl = ds_ctrl.isel({c:ctrl_years_mask})
        ds_hist = ds_hist.isel({c:hist_years_mask})
    
        ds_ctrl = ds_ctrl.assign_coords({c: ds_hist[c], f"{c}_original": time_ctrl.isel({c:ctrl_years_mask})})
    return ds_ctrl, ds_hist