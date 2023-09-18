import numpy as np
import xarray as xr
import gfdl_utils.core as gu

from .grid_preprocess import *

pre = "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_"
pp_dict = {
    "odiv-230": f"{pre}20221223/CM4_piControl_c192_OM4p25_v8/gfdl.ncrc4-intel18-prod-openmp/pp",
    "odiv-231": f"{pre}20221223/CM4_historical_c192_OM4p25/gfdl.ncrc4-intel18-prod-openmp/pp",
    "odiv-232": f"{pre}20221223/CM4_ssp585_c192_OM4p25/gfdl.ncrc4-intel18-prod-openmp/pp",
    "odiv-209": f"{pre}20210706/CM4_piControl_c192_OM4p125_v7/gfdl.ncrc4-intel18-prod-openmp/pp",
    "odiv-255": f"{pre}20230608/CM4_historical_c192_OM4p125/gfdl.ncrc5-intel22-prod-openmp/pp",
    "odiv-293": f"{pre}20230608/CM4_ssp585_c192_OM4p125/gfdl.ncrc5-intel22-prod-openmp/pp"
}

pre = "/archive/Raphael.Dussin/datasets/"
sg_dict = {
    "OM4p25": f"{pre}OM4p25/c192_OM4_025_grid_No_mg_drag_v20160808_unpacked/ocean_hgrid.nc",
    "OM4p125": f"{pre}OM4p125/mosaic_c192_om4p125_bedmachine_v20210310_hydrographyKDunne20210614_unpacked/ocean_hgrid.nc"
}

def get_pathDict(run, time="*", add="*", snap=False, surface=False):
    pp = pp_dict[run]
    suff1 = "ly" if "OM4p125" not in pp and not(snap) else ""
    suff2 = "_d2" if "OM4p125" in pp else ""
    suff2 += "_snap" if snap else "" 
    return {
        "pp": pp,
        "ppname": f"ocean_month{suff1}_z{suff2}",
        "out": "ts",
        "local": "monthly/5yr",
        "time": time,
        "add": add
    } if not(surface) else {
        "pp": pp,
        "ppname": f"ocean_monthly",
        "out": "ts",
        "local": "monthly/5yr",
        "time": time,
        "add": add
    }

def load_averages_and_snapshots(run, time="*"):
    averages = xr.merge([
        gu.open_frompp(**get_pathDict(run, time=time), chunks={'time':1}),
        gu.open_frompp(**get_pathDict(run, time=time, add=["*tos*", "*sos*"], surface=True), chunks={'time':1})
    ])
    snapshots = gu.open_frompp(**get_pathDict(run, time=time, snap=True), chunks={'time':1})
    snapshots = snapshots.rename({
        **{'time':'time_bounds'},
        **{v:f"{v}_bounds" for v in snapshots.data_vars}
    })
    return xr.merge([averages, snapshots])

def load_CM4highres_diags(testing=False):
    time =      "185001*" if testing else "*"
    time_ctrl = "010101*" if testing else "*"
    
    # Load mass/heat/salt budget diagnostics align times
    ctrl = load_averages_and_snapshots("odiv-230", time=time_ctrl)
    hist = load_averages_and_snapshots("odiv-231", time=time)
    if testing:
        ssp5 = hist
    else:
        ssp5 = load_averages_and_snapshots("odiv-232", time=time)
        ssp5 = xr.merge([
            xr.concat([hist.drop_dims('time'),        ssp5.drop_dims('time')], dim="time_bounds", combine_attrs="override"),
            xr.concat([hist.drop_dims('time_bounds'), ssp5.drop_dims('time_bounds')], dim="time", combine_attrs="override"),
        ], combine_attrs="override")

    # Align dates of control simulation with forced experiments that branch from it
    if testing:
        pass
    else:
        # Control is longer than forced experiments for some reason
        ctrl = ctrl.sel(time=ctrl.time[:-120], time_bounds=ctrl.time_bounds[:-120])
        
    ctrl_times, ctrl_time_bounds = ctrl.time.values.copy(), ctrl.time_bounds.values.copy()
    ctrl = ctrl.assign_coords({
        'time': xr.DataArray(ssp5.time.values, dims=("time",)),
        'time_bounds': xr.DataArray(ssp5.time_bounds.values, dims=("time_bounds",)),
    })
    # Keep record of original control dates for reference
    ctrl = ctrl.assign_coords({
        'time_original': xr.DataArray(ctrl_times, dims=("time",)),
        'time_bounds_original': xr.DataArray(ctrl_time_bounds, dims=("time_bounds",)),
    })
    
    ds = xr.concat([
        ssp5.expand_dims({'exp': ["forced"]}),
        ctrl.expand_dims({'exp': ["control"]})
    ], dim="exp", combine_attrs="override")
    
    
    if testing:
        # Only keep second fully year, to keep data load light
        ds = ds.isel(time=slice(12,24), time_bounds=slice(11, 24))
    else:
        # Get rid of first year since we don't have the initial snapshot of water masses
        ds = ds.isel(time=slice(12,None), time_bounds=slice(11,None))
    
    path_dict = get_pathDict("odiv-230")
    og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], path_dict["ppname"]))
    sg = xr.open_dataset(sg_dict["OM4p25" if "OM4p25" in path_dict["pp"] else "OM4p125"])

    og = fix_geo_coords(og, sg)
    ds = add_grid_coords(ds, og)
    grid = ds_to_grid(ds)
    
    return grid