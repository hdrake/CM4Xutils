import numpy as np
import dask
import xarray as xr
import xwmt
import xgcm
import doralite
import gfdl_utils.core as gu
import cftime

from .grid_preprocess import *
from .coarsen import *

exp_dict = {
    "CM4Xp25": {
        "hgrid": (
            "/archive/Raphael.Dussin/datasets/OM4p25/"
            "c192_OM4_025_grid_No_mg_drag_v20160808_unpacked/"
            "ocean_hgrid.nc"
        ),
        "piControl-spinup"   : "odiv-210",
        "piControl"          : "odiv-230",
        "piControl-continued": "odiv-306",
        "historical"         : "odiv-231",
        "ssp585"             : "odiv-232"
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

pre_pp = "/archive/Raphael.Dussin/FMS2019.01.03_devgfdl_"
pp_dict = {
    "CM4Xp25": {
        "hgrid": (
            "/archive/Raphael.Dussin/datasets/OM4p25/"
            "c192_OM4_025_grid_No_mg_drag_v20160808_unpacked/"
            "ocean_hgrid.nc"
        ),
        "piControl-spinup"   : f"{pre_pp}20210706/CM4_piControl_c192_OM4p25_v7/gfdl.ncrc4-intel18-prod-openmp/pp",
        "piControl"          : f"{pre_pp}20221223/CM4_piControl_c192_OM4p25_v8/gfdl.ncrc4-intel18-prod-openmp/pp",
        "piControl-continued": f"{pre_pp}20241030/CM4_piControl_c192_OM4p125_v8followup/gfdl.ncrc5-intel22-prod-openmp/pp",
        "historical"         : f"{pre_pp}20221223/CM4_historical_c192_OM4p25/gfdl.ncrc4-intel18-prod-openmp/pp",
        "ssp585"             : f"{pre_pp}20221223/CM4_ssp585_c192_OM4p25/gfdl.ncrc4-intel18-prod-openmp/pp",
    },
    "CM4Xp125": {
        "hgrid": (
            "/archive/Raphael.Dussin/datasets/OM4p125"
            "/mosaic_c192_om4p125_bedmachine_v20210310_hydrographyKDunne20210614_unpacked/"
            "ocean_hgrid.nc"
        ),
        "piControl-spinup": f"{pre_pp}20210706/CM4_piControl_c192_OM4p125_v7/gfdl.ncrc4-intel18-prod-openmp/pp",
        "piControl"       : f"{pre_pp}20230608/CM4_piControl_c192_OM4p125_v8/gfdl.ncrc5-intel22-prod-openmp/pp",
        "historical"      : f"{pre_pp}20230608/CM4_historical_c192_OM4p125/gfdl.ncrc5-intel22-prod-openmp/pp",
        "ssp585"          : f"{pre_pp}20230608/CM4_ssp585_c192_OM4p125/gfdl.ncrc5-intel22-prod-openmp/pp",
    }
}

def get_wmt_pathDict(model, exp, category, time="*", add="*"):
    """Retrieve dictionary of keyword arguments for `gfdl_utils.core.open_frompp`."""
    try:
        pp = doralite.dora_metadata(exp_dict[model][exp])['pathPP']
    except:
        print("Dora seems to be down. Using hard-coded paths instead.")
        pp = pp_dict[model][exp]
    freq = ["month"]
    ignore = ["1x1deg"]
    coarsen = ["d2"] if model=="CM4Xp125" else []
    if category=="surface":
        ignore_surf = ignore + coarsen if model=="CM4Xp125" else ignore
        ppname = gu.find_unique_variable(pp, "tos", require=freq, ignore=ignore_surf)
    elif category=="tendency":
        ppname = gu.find_unique_variable(pp, "opottemptend", require=freq+coarsen, ignore=ignore)
    elif category=="snapshot":
        ppname = gu.find_unique_variable(pp, "thetao", require=freq+["snap"]+coarsen, ignore=ignore)
    elif category=="ice":
        ppname = "ice"
    else:
        raise ValueError("Valid categories are 'surface', 'tendency', 'snapshot', and 'ice'.")
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
    """Load time-averaged water mass transformation budget diags and bounding snapshots"""
    pdict_tend = get_wmt_pathDict(model, exp, "tendency", time=time)
    av_tend = gu.open_frompp(**pdict_tend, dmget=dmget, mirror=mirror)

    # Derive shortwave flux convergence from shortwave fluxes
    if "rsdo" in av_tend.data_vars:
        vgrid = Grid(
            av_tend,
            coords={"Z": {"center":"z_l", "outer":"z_i"}},
            boundary={"Z":"extend"},
            autoparse_metadata=False
        )
        av_tend["rsdoabsorb"] = -vgrid.diff(av_tend.rsdo.chunk({"z_i":-1}), "Z")
        av_tend["rsdoabsorb"].attrs = {
            'cell_measures': 'volume: volcello area: areacello',
            'cell_methods': 'area:mean z_l:sum yh:mean xh:mean time:mean',
            'long_name': 'Convergence of Penetrative Shortwave Flux in Sea Water Layer',
            'standard_name': 'net_rate_of_absorption_of_shortwave_energy_in_ocean_layer',
            'time_avg_info': 'average_T1,average_T2,average_DT',
            'units': 'W m-2'
        }
    else:
        print(f"Missing `rsdo` diagnostic for {model}-{exp}")
    
    state_vars = ["tos", "sos"]
    mass_fluxes = ["wfo", "prlq", "prsn", "evs", "fsitherm", "friver", "ficeberg", "vprec"]
    mome_fluxes = ["taux", "tauy"]
    heat_fluxes = ["hflso", "hfsso", "rlntds", "heat_content_surfwater"]
    salt_fluxes = ["sfdsi"]
    surf_vars = state_vars + mass_fluxes + mome_fluxes + heat_fluxes + salt_fluxes
    pdict_surf = get_wmt_pathDict(model, exp, "surface" , time=time, add=surf_vars)
    av_surf = gu.open_frompp(**pdict_surf, dmget=dmget, mirror=mirror)

    # Interpolate wind stress to tracer points for simplicity
    hcoords = {
        "X": {'center':'xh', 'outer':'xq'},
        "Y": {'center':'yh', 'outer':'yq'}
    }
    hgrid = Grid(
        av_surf,
        coords=hcoords,
        boundary={"X":"periodic", "Y":"extend"},
        autoparse_metadata=False
    )
    if 'taux' in hgrid._ds.data_vars:
        av_surf['taux'] = hgrid.interp(hgrid._ds['taux'].chunk({"xq":-1}), 'X', keep_attrs=True)
        av_surf['taux'].attrs['cell_methods'] = 'yh:mean xh:mean time:mean'
    if 'tauy' in hgrid._ds.data_vars:
        av_surf['tauy'] = hgrid.interp(hgrid._ds['tauy'].chunk({"yq":-1}), 'Y', keep_attrs=True)
        av_surf['tauy'].attrs['cell_methods'] = 'yh:mean xh:mean time:mean'

    # For CM4Xp125, surface fluxes are only available on native grid,
    # but 3D tendencies only available on d2 coarsened grid,
    # so we need to coarsen fluxes to d2.
    if model == "CM4Xp125":
        og = xr.open_dataset(gu.get_pathstatic(pdict_surf["pp"], pdict_surf["ppname"]))
        correct_cell_methods(og)
        av_surf = add_grid_coords(av_surf, og)
        coords = {
            "X": {'center':'xh', 'outer':'xq'},
            "Y": {'center':'yh', 'outer':'yq'}
        }
        grid_tmp = Grid(
            av_surf,
            coords=coords,
            metrics={('X','Y'): "areacello"},
            boundary={"X":"periodic", "Y":"extend"},
            autoparse_metadata=False
        )
        av_surf = horizontally_coarsen(
            av_surf,
            grid_tmp,
            {"X":2, "Y":2}
        )
        av_surf = av_surf.assign_coords(av_tend.coords)
    
    ice_vars = ["siconc", "sithick", "LSNK", "LSRC", "EVAP", "SNOWFL", "RAIN"]
    pdict_ice = get_wmt_pathDict(model, exp, "ice" , time=time, add=ice_vars)
    av_ice = gu.open_frompp(**pdict_ice, dmget=dmget, mirror=mirror)
    av_ice = av_ice.drop_dims(
        [d for d in av_ice.dims if d not in ["time", "yT", "xT"]]
    )
    av_ice = av_ice.rename({"xT":"xh_ice", "yT":"yh_ice"})
    av_ice = av_ice.assign_coords({"time":av_tend.time})
    
    averages = xr.merge([av_tend, av_surf, av_ice]).chunk(chunk)

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
        elif (model == "CM4Xp25") and interval_preceeding in ["0356"]:
            pdict_snap_preceeding = get_wmt_pathDict(
                model, "piControl", "snapshot", time=f"035601*"
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

    ds_merged = xr.merge([averages, snapshots])
    return ds_merged

def load_wmt_grid(model, **kwargs):
    """Call `load_wmt_ds(model, **kwargs)` and build its corresponding `xgcm.Grid`."""
    ds = load_wmt_ds(model, **kwargs)
    grid = make_wmt_grid(ds)
    expand_surface_fluxes(grid)
    return grid

def expand_surface_fluxes(grid):
    mass_fluxes = ["wfo", "prlq", "prsn", "evs", "fsitherm", "friver", "ficeberg", "vprec"]
    heat_fluxes = ["hflso", "hfsso", "rlntds", "heat_content_surfwater"]
    salt_fluxes = ["sfdsi"]
    sice_fluxes = ["EVAP", "LSNK", "LSRC", "RAIN", "SNOWFL"]
    surf_fluxes = mass_fluxes + heat_fluxes + salt_fluxes + sice_fluxes
    
    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        wm = xwmt.WaterMass(grid)
        for v in surf_fluxes:
            attrs = grid._ds[v].attrs.copy()
            grid._ds[v] = (
                wm.expand_surface_array_vertically(grid._ds[v].fillna(0.), target_position="center")
                .transpose("exp", "time", "z_l", "yh", "xh")
            )
            attrs["cell_methods"] = "area:mean z_l:sum yh:mean xh:mean time: mean"
            attrs["long_name"] = f"Convergence of {attrs["long_name"]}"
            grid._ds[v].attrs = attrs

def load_wmt_ds(model, test=False, dmget=False, mirror=False, interval="all"):
    """Load a comprehensive CM4X dataset with all variables required to run `xwmb`."""
    if test:
        time =      "201001*"
        time_ctrl = "026101*"
        interval  = "2010"
        load_spin = False
        load_ctrl = True
        load_ctrl_continued = False
        load_hist = True
        load_ssp5 = False
    elif interval=="all":
        time = "*"
        time_ctrl = "*"
        load_spin = True
        load_ctrl = True
        load_ctrl_continued = True
        load_hist = True
        load_ssp5 = True
    elif interval.isnumeric():
        if (int(interval)%5)==0:
            time = f"{interval}01*"
            interval_ctrl = str(int(interval)-1749)
            time_ctrl = f"{interval_ctrl.zfill(4)}01*"
            load_spin = int(interval) < 1850
            load_ctrl = 1850 <= int(interval)
            if model == "CM4Xp25":
                load_ctrl = (1850 <= int(interval)) & (int(interval_ctrl) < 361)
            load_ctrl_continued = (model == "CM4Xp25") & (int(interval_ctrl) >= 361)
            load_hist = (1850 <= int(interval)) & (int(interval) < 2015)
            load_ssp5 = (2015 <= int(interval)) & (int(interval) < 2100)
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
        
    if load_ctrl_continued:
        print(f"Loading {model}-piControl-continued for interval `{interval}`.")
        ctrl_continued = load_wmt_averages_and_snapshots(
            model,
            "piControl-continued",
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

    if load_ctrl_continued:
        if load_ctrl:
            ctrl = concat_scenarios(ctrl, ctrl_continued)
        else:
            ctrl = ctrl_continued
    
    # Case 1: We are only loading in intervals from the spinup
    if load_spin and not(load_ctrl):
        ds = xr.concat([
            spinup.expand_dims({'exp': ["forced"]}),
            spinup.expand_dims({'exp': ["control"]})
        ], dim="exp", combine_attrs="override")

    # Case 2: We are loading in intervals from the control period (including forced runs)
    elif load_ctrl | load_ctrl_continued:
        if (load_hist) | (load_ssp5):
            ctrl, forc = align_dates(ctrl, forc)
            ds = xr.concat([
                forc.expand_dims({'exp': ["forced"]}),
                ctrl.expand_dims({'exp': ["control"]})
            ], dim="exp", combine_attrs="override")
        else:
            ds = ctrl.expand_dims({'exp': ["control"]})
        
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

def load_tracer(odiv, tracer, time="*"):
    """Load a CM4X dataset with `tracer`."""
    meta = doralite.dora_metadata(odiv)
    pp = meta['pathPP']
    ppname = "ocean_inert_z"
    if tracer == "agessc":
        ppname = "ocean_annual_z_d2" if "p125" in meta["expName"] else "ocean_annual_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    if (local.split("/")[1] == "5yr") or (time=="*"):
        ds = gu.open_frompp(
                pp, ppname, out, local, time, tracer,
                dmget=True
            )
    elif (local.split("/")[0] == "annual") and (local.split("/")[1] == "10yr"):
        if (int(time[:-1]) % 10) in [0,1]:
            ds = gu.open_frompp(
                pp, ppname, out, local, time, tracer,
                dmget=True
            ).isel(time=np.arange(0, 5, 1))
        else:
            ds = gu.open_frompp(
                pp, ppname, out, local, str(int(time[:-1]) - 5).zfill(4) + "*", tracer,
                dmget=True
            ).isel(time=np.arange(5, 10, 1))
  
    ds = ds.chunk({"time":1, "z_l":-1})
    
    return ds

def load_density(odiv, time="*"):
    """Load a CM4X dataset thermodynamics variables and derive sigma2."""
    state_vars = ["thkcello", "thetao", "so"]
    meta = doralite.dora_metadata(odiv)
    pp = meta['pathPP']
    ppname = "ocean_month_z" if "p125" in meta["expName"] else "ocean_monthly_z"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(
        pp, ppname, out, local, time, state_vars,
        dmget=True
    )
    ds = ds.chunk({"time":1, "z_l":-1})
    
    c_attrs = {c:ds.coords[c].attrs.copy() for c in ds.coords}

    CM4X_z_i_levels = np.array([
        0.000e+00, 5.000e+00, 1.500e+01, 2.500e+01, 4.000e+01, 6.250e+01,
        8.750e+01, 1.125e+02, 1.375e+02, 1.750e+02, 2.250e+02, 2.750e+02,
        3.500e+02, 4.500e+02, 5.500e+02, 6.500e+02, 7.500e+02, 8.500e+02,
        9.500e+02, 1.050e+03, 1.150e+03, 1.250e+03, 1.350e+03, 1.450e+03,
        1.625e+03, 1.875e+03, 2.250e+03, 2.750e+03, 3.250e+03, 3.750e+03,
        4.250e+03, 4.750e+03, 5.250e+03, 5.750e+03, 6.250e+03, 6.750e+03
    ])

    if "z_i" not in ds.coords:
        ds = ds.assign_coords({"z_i": xr.DataArray(
            CM4X_z_i_levels, dims=("z_i",), attrs = {
                'long_name': 'Depth at interface',
                'units': 'meters',
                'axis': 'Z',
                'positive': 'down'
            }
        )})
        
    og = gu.open_static(pp, ppname)
    model = [e for e,d in exp_dict.items() for k,v in d.items() if odiv==v][0]
    sg = xr.open_dataset(exp_dict[model]["hgrid"])
    og = fix_geo_coords(og, sg)
    ds = add_grid_coords(ds, og)

    # Compute potential density variables
    coords = {'Z': {'center': 'z_l', 'outer': 'z_i'}}
    wm_kwargs = {"coords": coords, "metrics":{}, "boundary":{"Z":"extend"}, "autoparse_metadata":False}
    wm_averages = xwmt.WaterMass(xgcm.Grid(ds[["thetao", "so", "thkcello", "z_i"]], **wm_kwargs))
    ds["sigma2"] = wm_averages.get_density("sigma2")

    for (c,a) in c_attrs.items():
        if not hasattr(ds.coords[c], 'attrs'):
            ds.coords[c].attrs = a
        else:
            for (k,v) in a.items():
                if k not in ds.coords[c].attrs.keys():
                    ds.coords[c].attrs[k] = v

    correct_cell_methods(ds)

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

def load_transient_tracers(odiv, time="*"):
    """Load all transient biogeochemical tracers and density."""
    transient_tracers = ["cfc11", "cfc12", "sf6"]
    try:
        ds_transient_tracers = load_tracer(odiv, transient_tracers, time=time)
    except:
        print(f"No transient tracers for {odiv}")
        ds_transient_tracers = xr.Dataset()
    ds_thickness = load_density(odiv, time=time)
    ds_transient_tracers = ds_transient_tracers.assign_coords({k:ds_thickness[k] for k in ds_thickness.coords})
    ds = xr.merge([ds_transient_tracers, ds_thickness], compat="override")

    grid = ds_to_grid(ds, Zprefix="z")

    return grid

def regrid_ice(ds, og, ig):
    if "xh_ice" in ds.coords:
        ds_ice = ds.drop_dims(["xh", "yh"])
        ds_ice = ds_ice.rename({"xh_ice":"xh", "yh_ice":"yh"})
        if ds.xh_ice.size == 2*ds.xh.size:
            ds_ice = ds_ice.assign_coords({
                "areacello": xr.DataArray(
                    ig.CELL_AREA.values, dims=("yh", "xh"), attrs=og.wet.attrs
                ),
                "wet": xr.DataArray(
                    og.wet.values, dims=("yh", "xh"), attrs=og.wet.attrs
                ),
            })
            grid_ice = Grid(
                ds_ice,
                coords={"X":{'center':'xh'},"Y": {'center':'yh'}},
                metrics={('X','Y'): "areacello"},
                boundary={"X":"periodic", "Y":"extend"},
                autoparse_metadata=False
            )
            ds_ice = horizontally_coarsen(
                ds_ice,
                grid_ice,
                {"X":2, "Y":2},
                skip_coords=True
            ).drop_vars(["areacello", "wet"])
        ds = xr.merge([
            ds.drop_dims(["xh_ice", "yh_ice"]),
            ds_ice.assign_coords({"xh":ds.xh, "yh":ds.yh})
        ])
    return ds
            

def make_wmt_grid(ds, overwrite_grid=True, overwrite_supergrid=True):
    """Make a comprehensive `xwmb`-compatible `xgcm.Grid` object."""

    c_attrs = {
        c:ds.coords[c].attrs.copy() for c in ds.coords
        if c not in ["xh_ice", "yh_ice"]
    }

    if overwrite_grid:
        path_dict = get_wmt_pathDict(ds.attrs["model"], "piControl", "surface")

        if ds.attrs["model"] == "CM4Xp125":
            # Correct d2 coordinates by starting from full-resolution static file / supergrid
            og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], "ocean_annual"))
        else:
            og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], path_dict["ppname"]))
            
        if overwrite_supergrid:
            print(f"Overriding {ds.attrs['model']} grid coordinates from supergrid.")
            sg = xr.open_dataset(exp_dict[ds.attrs["model"]]["hgrid"])
            og = fix_geo_coords(og, sg)

        # Correct d2 coordinates by manually coarsening full-resolution static file
        if ds.attrs["model"] == "CM4Xp125":
            og = add_grid_coords(og, og)
            og = og.drop_vars(og.data_vars)
            correct_cell_methods(og)
            coords = {
                "X": {'center':'xh', 'outer':'xq'},
                "Y": {'center':'yh', 'outer':'yq'}
            }
            grid_full = Grid(
                og,
                coords=coords,
                metrics={('X','Y'): "areacello"},
                boundary={"X":"periodic", "Y":"extend"},
                autoparse_metadata=False
            )
            og = horizontally_coarsen(
                og,
                grid_full,
                {"X":2, "Y":2}
            )

        ds = add_grid_coords(ds, og)

        # Add cell_methods to variables if missing (e.g. for sea ice fields)
        for v in [v for v in ds.data_vars if "xh_ice" in ds[v].dims]:
            if "cell_methods" in ds[v].attrs:
                if ds[v].ndim == 4:
                    ds[v].attrs["cell_methods"] = 'area:mean yh:mean xh:mean time: mean'
                else:
                    ds = ds.drop(v)
    
        # Regrid
        print("Regridding ice")
        og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], "ocean_annual"))
        og = add_grid_coords(og, og)
        og = og.drop_vars(og.data_vars)
        correct_cell_methods(og)
        ig = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], "ice"))
        ds = regrid_ice(ds, og, ig)
    
    grid = ds_to_grid(ds)

    # Correct fsitherm and prlq ocean flux diagnostics using RAIN ice diagnostic
    if all([e in ds.data_vars for e in ["prlq", "RAIN"]]):
        ds["fsitherm"].data = ds["prlq"].data - ds["RAIN"].data
        if "fsitherm" in ds.data_vars:
            ds["prlq"].data = ds["RAIN"].data

    # Construct 3D h_tendency from wfo if it does not exist
    if "boundary_forcing_h_tendency" not in grid._ds:
        with dask.config.set(**{'array.slicing.split_large_chunks': False}):
            wm = xwmt.WaterMass(grid)
            grid._ds["boundary_forcing_h_tendency"] = (
                -grid.diff(wm.expand_surface_array_vertically(grid._ds["wfo"]), axis="Z") /
                1035.
            ).rename("boundary_forcing_h_tendency")
            grid._ds["boundary_forcing_h_tendency"].attrs = {
                'long_name': 'Cell thickness tendency due to boundary forcing',
                'units': 'm s-1',
                'cell_methods': 'area:mean z_l:sum yh:mean xh:mean time: mean',
                'cell_measures': 'volume: volcello area: areacello',
                'time_avg_info': 'average_T1,average_T2,average_DT'
            }
    
    # Compute potential density variables
    coords = {'Z': grid.axes['Z'].coords}
    wm_kwargs = {"coords": coords, "metrics":{}, "boundary":{"Z":"extend"}, "autoparse_metadata":False}
    wm_averages = xwmt.WaterMass(xgcm.Grid(grid._ds[["thetao", "so", "thkcello", coords["Z"]["outer"]]], **wm_kwargs))
    grid._ds["sigma2"] = wm_averages.get_density("sigma2")
    snapshot_state_vars = grid._ds[["thetao_bounds", "so_bounds", "thkcello_bounds", coords["Z"]["outer"]]]
    rename_vardict = {v:v.split("_")[0] for v in snapshot_state_vars.data_vars}
    wm_snapshots = xwmt.WaterMass(xgcm.Grid(snapshot_state_vars.rename(rename_vardict), **wm_kwargs))
    grid._ds["sigma2_bounds"] = wm_snapshots.get_density("sigma2")

    for (c,a) in c_attrs.items():
        if not hasattr(grid._ds.coords[c], 'attrs'):
            grid._ds.coords[c].attrs = a
        else:
            for (k,v) in a.items():
                if k not in grid._ds.coords[c].attrs.keys():
                    grid._ds.coords[c].attrs[k] = v
    
    return grid

def concat_scenarios(ds_list):
    """Concatinate scenarios in list over all "time" dimensions."""
    return xr.merge([
        xr.concat([
            ds.drop_dims([dim for dim in ds.dims if (dim!=cdim) & ("time" in dim)])
            for ds in ds_list
        ], dim=cdim, combine_attrs="override")
        for cdim in ds_list[0].dims if "time" in cdim
    ], combine_attrs="override")

def align_dates(ds_ctrl, ds_hist):
    """Align dates of CM4X piControl and forced experiments
    
    Parameters
    ----------
    ds_ctrl : piControl or piControl_spinup simulation (starting in 0001)
    ds_hist : historical or other forced simulation (branching as 1850 in 0101)

    Returns
    -------
    (ds_ctrl, ds_hist) : input datasets with modified time coordinates
    
    """
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
    if "time_bounds" in ds_hist.coords:
        ds_hist.coords["time_bounds"].attrs = {
            **ds_ctrl.coords["time_bounds_since_init"].attrs,
            **{"long_name": "historical time", "cell_methods": "time:point"}
        }

    ds_ctrl.coords["time_since_init"].attrs = {
        **ds_ctrl.coords["time_since_init"].attrs,
        **{"long_name": "time since model initialization", "cell_methods": "time:mean"}
    }
    if "time_bounds_since_init" in ds_ctrl.coords:
        ds_ctrl.coords["time_bounds_since_init"].attrs = {
            **ds_ctrl.coords["time_bounds_since_init"].attrs,
            **{"long_name": "time since model initialization", "cell_methods": "time:point"}
        }
        
    return ds_ctrl, ds_hist
