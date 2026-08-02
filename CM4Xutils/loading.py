"""GFDL-specific loaders for CM4X post-processed diagnostics.

These functions only run on GFDL analysis machines: they depend on `doralite`
(experiment metadata), `gfdl_utils` (``open_frompp`` / static-file access),
`/archive` paths, and ``dmget`` tape retrieval.

`exp_dict` / `pp_dict` map (model, experiment) to Dora IDs and hard-coded ``pp``
paths. The high-level entry point is `load_wmt_grid` -> `load_wmt_ds` ->
`load_wmt_averages_and_snapshots`, which assembles time-mean budget tendencies,
surface fluxes, sea-ice diagnostics, and bounding snapshots into a single dataset
carrying both a `time` (monthly-mean) and `time_bounds` (snapshot) axis, and a
`control`/`forced` `exp` dimension. See CLAUDE.md for the full pipeline.
"""

import os
import glob
import numpy as np
import dask
import xarray as xr
import xwmt
import xeos
import xgcm
import doralite
import gfdl_utils.core as gu
import cftime

from .grid_preprocess import *
from .coarsen import *

# The CM4X MOM6 runs were configured with EQN_OF_STATE="WRIGHT" (verified from the
# MOM_parameter_doc.all of both CM4Xp25 and CM4Xp125). We compute sigma2 offline with the
# *same* equation of state via xeos, rather than gsw/TEOS-10 (which the model never used).
#
# xeos "wright97-reduced" has byte-identical coefficients and an identical density formula
# to MOM6's legacy WRIGHT kernel; the two differ only in floating-point addition
# associativity (~1e-12 kg/m3), so this matches the online density to machine precision.
# (MOM6's documented `use_Wright_2nd_deriv_bug` affects only second derivatives, not
# density or the alpha/beta first derivatives.) Once a bit-exact legacy kernel is vendored
# in xeos, this can become `xeos.from_model("MOM6", "WRIGHT")`.
CM4X_EOS = xeos.from_model("MOM6", "WRIGHT_REDUCED")


def cm4x_watermass(grid):
    """Build an `xwmt.WaterMass` configured with the CM4X model equation of state.

    `t_var` / `s_var` declare the *kinds* of the archived `thetao` and `so`, which is how
    `xwmt` decides whether to convert them before evaluating `CM4X_EOS`. Declaring them
    potential/practical -- the kinds MOM6's Wright EOS consumes -- makes that conversion a
    no-op, so `thetao`/`so` go straight into Wright. That is bit-for-bit the operation
    MOM6 performs online, so the offline sigma2 reproduces the model's own density.

    Two independent checks confirm potential/practical is the correct declaration here:

    - MOM6's `USE_CONTEMP_ABSSAL` is the switch that declares the prognostic tracers to be
      conservative temperature and absolute salinity (`MOM.F90` sets `tv%T_is_conT` /
      `tv%S_is_absS` from it, which relabel the T/S diagnostics and trigger gsw
      conversions in `MOM_EOS.F90`). Both CM4Xp25 and CM4Xp125 ran with
      `USE_CONTEMP_ABSSAL = False`, i.e. the model itself declares its T/S to be potential
      temperature and practical salinity.
    - The archived diagnostics agree: `thetao` carries
      `standard_name = "sea_water_potential_temperature"` and `so` is in psu.

    This is a statement about the model's internal convention, not about which variable is
    the most physically faithful. McDougall et al. (2021) argue that a model conserving
    heat as `Cp*T` with constant `Cp` is better interpreted as carrying conservative
    temperature, and that reading is why `xwmt` defaults to
    `t_var="conservative"` / `s_var="absolute"`. But that argument is about how to compare
    MOM6 output to observations; it does not change what MOM6 actually computed. Since the
    purpose here is a sigma2 consistent with the model's own dynamics -- the density that
    set the isopycnal structure and the water-mass transformation being diagnosed -- we
    reproduce the model's arithmetic exactly.
    """
    return xwmt.WaterMass(grid, eos=CM4X_EOS, t_var="potential", s_var="practical")


def _restore_coord_attrs(ds, c_attrs):
    """Re-attach coordinate attributes that xgcm/xarray operations dropped.

    Only keys that are *missing* from the current attributes are restored, so
    attributes deliberately (re)written downstream -- e.g. the corrected
    ``cell_methods`` from `correct_cell_methods` -- always win. Coordinates that no
    longer exist on `ds` are skipped rather than raising.

    Parameters
    ----------
    ds : `xr.Dataset` to patch in place
    c_attrs : dict, coord name -> attribute dict captured before the operation
    """
    for (c, a) in c_attrs.items():
        if c not in ds.coords:
            continue
        for (k, v) in a.items():
            ds.coords[c].attrs.setdefault(k, v)


# The three loaders below stamp the same human-readable experiment description
# onto their output; keep it in one place so it cannot drift between them.
def _experiment_description(model):
    """Return the standard human-readable description of the CM4X experiment design."""
    return (
f"""The {model} experimental design following Griffies et al., published in the
Journal of Advances in Modeling Earth Systems (JAMES):
https://doi.org/10.1029/2024MS004861 and https://doi.org/10.1029/2024MS004862.

The `control` experiment type is a ocean-sea ice-atmosphere-land coupled
climate model run with CO2 concentrations in the atmosphere
prescribed at 280 ppm (preindustrial levels).

The `forced` experiment type branches off from the preindustrial control
in 1850 and is forced with historical CMIP6 forcings until 2014 and afterwards
follows the SSP5-8.5 high-emissions forcing scenario."""
    )

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
        "piControl-spinup"   : "odiv-209",
        "piControl"          : "odiv-313",
        "piControl-continued": "odiv-437",
        "historical"         : "odiv-255",
        "ssp585"             : "odiv-293",
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
        "piControl-continued": f"{pre_pp}20221223/CM4_piControl_c192_OM4p25_v8/gfdl.ncrc4-intel22-prod-openmp/pp",
        "historical"         : f"{pre_pp}20221223/CM4_historical_c192_OM4p25/gfdl.ncrc4-intel18-prod-openmp/pp",
        "ssp585"             : f"{pre_pp}20221223/CM4_ssp585_c192_OM4p25/gfdl.ncrc4-intel18-prod-openmp/pp",
    },
    "CM4Xp125": {
        "hgrid": (
            "/archive/Raphael.Dussin/datasets/OM4p125"
            "/mosaic_c192_om4p125_bedmachine_v20210310_hydrographyKDunne20210614_unpacked/"
            "ocean_hgrid.nc"
        ),
        "piControl-spinup"    : f"{pre_pp}20210706/CM4_piControl_c192_OM4p125_v7/gfdl.ncrc4-intel18-prod-openmp/pp",
        "piControl"           : f"{pre_pp}20230608/CM4_piControl_c192_OM4p125_v8/gfdl.ncrc5-intel22-prod-openmp/pp",
        "piControl-continued" : f"{pre_pp}20241030/CM4_piControl_c192_OM4p125_v8followup/gfdl.ncrc5-intel23-prod-openmp/pp",
        "historical"          : f"{pre_pp}20230608/CM4_historical_c192_OM4p125/gfdl.ncrc5-intel22-prod-openmp/pp",
        "ssp585"              : f"{pre_pp}20230608/CM4_ssp585_c192_OM4p125/gfdl.ncrc5-intel22-prod-openmp/pp",
    }
}

def get_wmt_pathDict(model, exp, category, time="*", add="*"):
    """Retrieve keyword arguments for `gfdl_utils.core.open_frompp`.

    Resolves the ``pp`` root via Dora (falling back to the hard-coded `pp_dict`
    paths if Dora is unreachable) and picks the ``ppname`` (pp component) that
    holds the requested `category` of diagnostics. For CM4Xp125, 3D categories are
    taken from the "d2" (2x-coarsened) components while surface fields stay native.

    Parameters
    ----------
    model : "CM4Xp25" or "CM4Xp125"
    exp : experiment name (key of `exp_dict[model]`)
    category : one of {"surface", "tendency", "native", "snapshot", "ice"}
    time : GFDL time glob passed through to `open_frompp`
    add : extra variable(s) to request alongside the category's probe variable

    Returns
    -------
    dict : keyword arguments (pp, ppname, out, local, time, add) for `open_frompp`
    """
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
    elif category=="native":
        ignore_list = ignore+coarsen+["snap"]+["rho2"]
        ppname = gu.find_unique_variable(pp, "thkcello", require=freq, ignore=ignore_list)
        add = "thkcello"
    elif category=="snapshot":
        ppname = gu.find_unique_variable(pp, "thetao", require=freq+["snap"]+coarsen, ignore=ignore)
    elif category=="ice":
        ppname = "ice"
    else:
        raise ValueError("Valid categories are 'surface', tendency', 'native', 'snapshot', and 'ice'.")
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
    """Load time-averaged WMT budget diagnostics and their bounding snapshots.

    Assembles, for a single (model, experiment):
      - time-mean heat/salt budget tendencies (+ derived shortwave convergence),
      - surface state variables and mass/heat/salt/momentum fluxes (wind stress is
        interpolated to tracer points; for CM4Xp125 all surface fields are coarsened
        by 2x to match the "d2" grid the tendencies live on),
      - sea-ice diagnostics (renamed onto their own xh_ice/yh_ice grid),
      - instantaneous snapshots renamed onto a `time_bounds` axis with a `_bounds`
        suffix, including the trailing snapshot of the preceding 5-year interval
        (with hard-coded corner cases at experiment branch points).

    Parameters
    ----------
    model : "CM4Xp25" or "CM4Xp125"
    exp : experiment name (e.g. "piControl", "historical", "ssp585")
    time : GFDL time glob (e.g. "*" for all, or "185001*" for one 5-year interval)
    dmget, mirror : forwarded to `gfdl_utils.core.open_frompp` for tape staging

    Returns
    -------
    ds_merged : `xr.Dataset` with both `time` (averages) and `time_bounds` (snapshots)
    """

    # Get time-averaged heat and salt budget tendencies
    pdict_tend = get_wmt_pathDict(model, exp, "tendency", time=time)
    av_tend = gu.open_frompp(**pdict_tend, dmget=dmget, mirror=mirror)

    # Derive shortwave flux convergence from shortwave fluxes
    if "rsdo" in av_tend.data_vars:
        vgrid = Grid(
            av_tend,
            coords={"Z": {"center":"z_l", "outer":"z_i"}},
            padding={"Z":"extend"},
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
        append_provenance(av_tend["rsdoabsorb"], (
            "Derived by CM4Xutils as minus the vertical difference of the archived "
            "downwelling shortwave flux `rsdo` across each layer's interfaces "
            "(extended at the top and bottom), i.e. the shortwave energy absorbed "
            "within the layer. Not an archived model diagnostic."
        ))
    else:
        print(f"Missing `rsdo` diagnostic for {model}-{exp}")
    
    state_vars = ["tos", "sos", "ePBL_h_ML", "zos"]
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
        padding={"X":"periodic", "Y":"extend"},
        autoparse_metadata=False
    )
    if 'taux' in hgrid._ds.data_vars:
        av_surf['taux'] = hgrid.interp(hgrid._ds['taux'].chunk({"xq":-1}), 'X', keep_attrs=True)
        av_surf['taux'].attrs['cell_methods'] = 'yh:mean xh:mean time:mean'
        append_provenance(av_surf['taux'], (
            "Interpolated from the zonal-velocity (Cu, `xq`) points onto the tracer "
            "cell centers (`xh`) as the two-point average of the faces bounding each "
            "cell in X (periodic in X). It is therefore no longer the value the model "
            "applied at the u-points."
        ))
    if 'tauy' in hgrid._ds.data_vars:
        av_surf['tauy'] = hgrid.interp(hgrid._ds['tauy'].chunk({"yq":-1}), 'Y', keep_attrs=True)
        av_surf['tauy'].attrs['cell_methods'] = 'yh:mean xh:mean time:mean'
        append_provenance(av_surf['tauy'], (
            "Interpolated from the meridional-velocity (Cv, `yq`) points onto the "
            "tracer cell centers (`yh`) as the two-point average of the faces "
            "bounding each cell in Y (edge values extended at the Y boundaries). It "
            "is therefore no longer the value the model applied at the v-points."
        ))

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
            padding={"X":"periodic", "Y":"extend"},
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
    
    averages = chunk_dataset(xr.merge([av_tend, av_surf, av_ice]), chunk)

    # Case 1: We are either reading in all times or just the first 5-year interval.
    # In either case, there is no prior 5 yr interval, so we're missing the initial snapshot.
    if (time=="*") | (time=="000101*"):
        pdict_snap = get_wmt_pathDict(model, exp, "snapshot", time=time)
        snapshots = chunk_dataset(gu.open_frompp(**pdict_snap, dmget=dmget, mirror=mirror), chunk_center)

    # Case 2: we are only reading in a specific 5 yr interval,
    # in which case we also need the last snapshot from the prior interval.
    elif (time!="*") & (time!="000101*"):
        interval_preceding = str(np.int64(time.split("*")[0][:-2]) - 5).zfill(4)
        # Awkward corner cases where the prior interval is actually from a different experiment
        if interval_preceding in ["0096", "1845"]:
            pdict_snap_preceding = get_wmt_pathDict(
                model, "piControl-spinup", "snapshot", time=f"009601*"
            )
        elif (model == "CM4Xp25") and interval_preceding in ["0356"]:
            pdict_snap_preceding = get_wmt_pathDict(
                model, "piControl", "snapshot", time=f"035601*"
            )
        elif (model == "CM4Xp125") and interval_preceding in ["0446"]:
            pdict_snap_preceding = get_wmt_pathDict(
                model, "piControl", "snapshot", time=f"044601*"
            )
        elif interval_preceding=="2010":
            pdict_snap_preceding = get_wmt_pathDict(
                model, "historical", "snapshot", time=f"201001*"
            )
        else:
            pdict_snap_preceding = get_wmt_pathDict(
                model, exp, "snapshot", time=f"{interval_preceding}01*"
            )
        pdict_snap = get_wmt_pathDict(model, exp, "snapshot", time=time)
        snapshots = xr.concat(
            [
                chunk_dataset(gu.open_frompp(**pdict_snap_preceding, dmget=dmget, mirror=mirror), chunk_center).isel(time=-1),# only the last
                chunk_dataset(gu.open_frompp(**pdict_snap, dmget=dmget, mirror=mirror), chunk_center)
            ],
            dim="time"
        )

    # Move snapshots onto their own `time_bounds` axis and suffix every snapshot
    # variable with `_bounds`, so time-mean and instantaneous fields can coexist
    # in one merged dataset without name collisions.
    snapshots = snapshots.rename({
        **{'time':'time_bounds'},
        **{v:f"{v}_bounds" for v in snapshots.data_vars}
    })
    # The snapshot stream inherits `time: mean` from the archived files even though
    # these are instantaneous values, which mislabels every `*_bounds` variable in
    # the released product as a time average. Only the X/Y/Z parts of `cell_methods`
    # are dispatched on, so correcting the time part changes no numbers.
    for v in snapshots.data_vars:
        if "cell_methods" in snapshots[v].attrs:
            cm = parse_cell_methods(snapshots[v].attrs["cell_methods"])
            if cm.get("time") == "mean":
                cm["time"] = "point"
                snapshots[v].attrs["cell_methods"] = stringify_cell_methods_dict(cm)
        # `time_avg_info` points at averaging-period variables that do not apply to
        # an instantaneous field.
        snapshots[v].attrs.pop("time_avg_info", None)

    ds_merged = xr.merge([averages, snapshots])
    return ds_merged

def load_wmt_grid(model, **kwargs):
    """Call `load_wmt_ds(model, **kwargs)` and build its corresponding `xgcm.Grid`."""
    ds = load_wmt_ds(model, **kwargs)
    grid = make_wmt_grid(ds)
    expand_surface_fluxes(grid)
    return grid

def expand_surface_fluxes(grid):
    """Recast 2D surface fluxes as 3D layer convergences, in place on `grid._ds`.

    `xwmb` treats surface boundary fluxes as a convergence into the topmost model
    layer. This inserts each 2D flux into the vertical grid at the surface (zeros
    elsewhere) via `xwmt.WaterMass.expand_surface_array_vertically`, and rewrites
    its `cell_methods`/`long_name` to reflect the new 3D "Convergence of ..." field.
    """
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
            attrs["long_name"] = f"Convergence of {attrs['long_name']}"
            grid._ds[v].attrs = attrs
            append_provenance(grid._ds[v], (
                "Recast from a 2D surface flux into a 3D layer convergence by "
                "inserting the whole flux into the topmost model layer and setting "
                "every deeper layer to zero, as `xwmb` expects boundary fluxes to be "
                "supplied. Values are unchanged; only the vertical placement is new."
            ))

def _interval_load_flags(model, interval, test=False):
    """Determine which experiments and time patterns to load for a given interval.

    Shared by `load_wmt_ds` and `load_rho2_transports_ds` so that both build the same
    control/forced `exp` structure and calendar alignment for a given `interval`.

    Returns
    -------
    dict with keys: "time", "time_ctrl", "interval", "load_spin", "load_ctrl",
    "load_ctrl_continued", "load_hist", "load_ssp5".
    """
    if test:
        return {
            "time": "201001*", "time_ctrl": "026101*", "interval": "2010",
            "load_spin": False, "load_ctrl": True, "load_ctrl_continued": False,
            "load_hist": True, "load_ssp5": False,
        }
    elif interval == "all":
        return {
            "time": "*", "time_ctrl": "*", "interval": interval,
            "load_spin": True, "load_ctrl": True, "load_ctrl_continued": True,
            "load_hist": True, "load_ssp5": True,
        }
    elif interval.isnumeric():
        if (int(interval) % 5) == 0:
            # Control years are offset by 1749 from the forced (historical) calendar.
            interval_ctrl = str(int(interval) - 1749)
            # The piControl was extended ("continued") from a later restart, so the
            # continued branch takes over once the control year reaches `continue_year`.
            continue_year = 361 if (model == "CM4Xp25") else 451
            return {
                "time": f"{interval}01*",
                "time_ctrl": f"{interval_ctrl.zfill(4)}01*",
                "interval": interval,
                "load_spin": int(interval) < 1850,
                "load_ctrl": (1850 <= int(interval)) & (int(interval_ctrl) < continue_year),
                "load_ctrl_continued": (int(interval_ctrl) >= continue_year),
                "load_hist": (1850 <= int(interval)) & (int(interval) < 2015),
                "load_ssp5": (2015 <= int(interval)) & (int(interval) < 2100),
            }
        else:
            raise ValueError("interval must be an integer multiple of 5.")
    else:
        raise ValueError(
            f"interval must be 'all' or a string of an integer multiple of 5, "
            f"got {interval!r}."
        )

def load_wmt_ds(model, test=False, dmget=False, mirror=False, interval="all"):
    """Load a comprehensive CM4X dataset with all variables required to run `xwmb`."""
    flags = _interval_load_flags(model, interval, test=test)
    time                = flags["time"]
    time_ctrl           = flags["time_ctrl"]
    interval            = flags["interval"]
    load_spin           = flags["load_spin"]
    load_ctrl           = flags["load_ctrl"]
    load_ctrl_continued = flags["load_ctrl_continued"]
    load_hist           = flags["load_hist"]
    load_ssp5           = flags["load_ssp5"]

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
        forc = concat_scenarios([hist, ssp5])
    elif load_hist:
        forc = hist
    elif load_ssp5:
        forc = ssp5
    
    # The historical only branches off after the spin-up, we need to expand the
    # exp dimension of the spinup and specify it for both the control and forced.

    if load_ctrl_continued:
        if load_ctrl:
            ctrl = concat_scenarios([ctrl, ctrl_continued])
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
            ds = concat_scenarios([spinup, ds])
    
    if test:
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # Only keep second full year, to keep data load light
            ds = ds.isel(time=slice(12,24), time_bounds=slice(11, 24))

    ds.coords["exp"].attrs = {"long_name": "Experiment type"}
    ds.attrs["model"] = model
    ds.attrs["description"] = _experiment_description(model)

    return ds

def _rho2_pp(model, exp):
    """Resolve the pp path for one experiment (Dora, with hard-coded fallback)."""
    try:
        return doralite.dora_metadata(exp_dict[model][exp])['pathPP']
    except:
        print("Dora seems to be down. Using hard-coded paths instead.")
        return pp_dict[model][exp]

def available_rho2_transports(model, exp):
    """Return which of {'umo','vmo'} are archived in `ocean_month_rho2`.

    CM4X does not archive the same density-coordinate transports for every model:
    CM4Xp125 saves both `umo` and `vmo`, but CM4Xp25 saves only `vmo`. We probe the
    filesystem so the budget pipeline adapts automatically rather than hard-coding this.
    """
    pp = _rho2_pp(model, exp)
    local = gu.get_local(pp, "ocean_month_rho2", "ts")
    base = os.path.join(pp, "ocean_month_rho2", "ts", local)
    return {v for v in ["umo", "vmo"] if glob.glob(os.path.join(base, f"*.{v}.nc"))}

def online_rho2_transport_vars(model, interval="all"):
    """Transport vars available online-remapped across *all* experiments in `interval`.

    Returns the intersection of `available_rho2_transports` over the experiments that
    `_interval_load_flags` selects, so the budget pipeline only sources a transport
    from `ocean_month_rho2` when it exists for every branch it needs to concatenate.
    """
    flags = _interval_load_flags(model, interval)
    exps = [e for (e, key) in [
        ("piControl-spinup",    "load_spin"),
        ("piControl",           "load_ctrl"),
        ("piControl-continued", "load_ctrl_continued"),
        ("historical",          "load_hist"),
        ("ssp585",              "load_ssp5"),
    ] if flags[key]]
    avail = None
    for e in exps:
        a = available_rho2_transports(model, e)
        avail = a if avail is None else (avail & a)
    return avail or set()

def load_rho2_transports(model, exp, time="*", dmget=False, mirror=False, transport_vars=None):
    """Load online-remapped density-coordinate mass transports for one experiment.

    MOM6 accumulates the layer-integrated transports `umo`/`vmo` (and `thkcello`)
    *online* into potential-density (`rho2`) layers and archives them in the
    `ocean_month_rho2` pp output, where they conserve mass exactly within each density
    layer. This is much more accurate than remapping z-coordinate transports offline, so
    the budget pipeline sources transports from here rather than from
    `remap_vertical_coord`. See `load_rho2` in the CM4XTransientTracers `preprocessing`
    module for the analogous single-experiment loader this follows.

    `transport_vars` selects which of {'umo','vmo'} to load; if None, whichever are
    archived (`available_rho2_transports`) are used. Returns an `xr.Dataset` (not a grid)
    on the `rho2_l` coordinate so it can be passed through `concat_scenarios` /
    `align_dates` by `load_rho2_transports_ds`.
    """
    pp = _rho2_pp(model, exp)
    ppname = "ocean_month_rho2"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    if transport_vars is None:
        transport_vars = available_rho2_transports(model, exp)
    load_vars = sorted(transport_vars) + ["thkcello"]
    ds = gu.open_frompp(
        pp, ppname, out, local, time, load_vars,
        dmget=dmget, mirror=mirror
    )
    ds = chunk_dataset(ds, {"time": 1, "rho2_l": -1})

    og = gu.open_static(pp, ppname)
    sg = xr.open_dataset(exp_dict[model]["hgrid"])
    og = fix_geo_coords(og, sg)
    ds = add_grid_coords(ds, og)

    ds.attrs["model"] = model
    return ds

def load_rho2_transports_ds(model, dmget=False, mirror=False, interval="all"):
    """Load online-remapped rho2 transports with the same structure as `load_wmt_ds`.

    Mirrors the experiment-selection, `exp`-dimension concatenation, and calendar
    alignment of `load_wmt_ds` (via the shared `_interval_load_flags` helper,
    `concat_scenarios`, and `align_dates`) so the returned transports can be merged
    directly into the budget product. Only transports available across *all* loaded
    experiments (`online_rho2_transport_vars`) are loaded, so the concatenation is
    consistent. Transports are time-means only, so none of the snapshot / `time_bounds`
    machinery of `load_wmt_ds` is needed here.
    """
    flags = _interval_load_flags(model, interval)
    time                = flags["time"]
    time_ctrl           = flags["time_ctrl"]
    load_spin           = flags["load_spin"]
    load_ctrl           = flags["load_ctrl"]
    load_ctrl_continued = flags["load_ctrl_continued"]
    load_hist           = flags["load_hist"]
    load_ssp5           = flags["load_ssp5"]

    tvars = online_rho2_transport_vars(model, interval)

    if load_spin:
        print(f"Loading {model}-piControl-spinup transports for interval `{interval}`.")
        spinup = load_rho2_transports(model, "piControl-spinup", time=time_ctrl, dmget=dmget, mirror=mirror, transport_vars=tvars)
    if load_ctrl:
        print(f"Loading {model}-piControl transports for interval `{interval}`.")
        ctrl = load_rho2_transports(model, "piControl", time=time_ctrl, dmget=dmget, mirror=mirror, transport_vars=tvars)
    if load_ctrl_continued:
        print(f"Loading {model}-piControl-continued transports for interval `{interval}`.")
        ctrl_continued = load_rho2_transports(model, "piControl-continued", time=time_ctrl, dmget=dmget, mirror=mirror, transport_vars=tvars)
    if load_hist:
        print(f"Loading {model}-historical transports for interval `{interval}`.")
        hist = load_rho2_transports(model, "historical", time=time, dmget=dmget, mirror=mirror, transport_vars=tvars)
    if load_ssp5:
        print(f"Loading {model}-ssp585 transports for interval `{interval}`.")
        ssp5 = load_rho2_transports(model, "ssp585", time=time, dmget=dmget, mirror=mirror, transport_vars=tvars)

    if load_hist and load_ssp5:
        forc = concat_scenarios([hist, ssp5])
    elif load_hist:
        forc = hist
    elif load_ssp5:
        forc = ssp5

    if load_ctrl_continued:
        if load_ctrl:
            ctrl = concat_scenarios([ctrl, ctrl_continued])
        else:
            ctrl = ctrl_continued

    # Case 1: only spinup intervals
    if load_spin and not(load_ctrl):
        ds = xr.concat([
            spinup.expand_dims({'exp': ["forced"]}),
            spinup.expand_dims({'exp': ["control"]})
        ], dim="exp", combine_attrs="override")

    # Case 2: control period (including forced runs)
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
            ds = concat_scenarios([spinup, ds])

    ds.coords["exp"].attrs = {"long_name": "Experiment type"}
    ds.attrs["model"] = model
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
    else:
        # Previously fell through with `ds` unbound, raising an opaque
        # UnboundLocalError further down.
        raise ValueError(
            f"Don't know how to slice a single interval out of pp chunking {local!r} "
            f"for {ppname}; only 'ts/*/5yr' and 'ts/annual/10yr' are handled."
        )

    ds = chunk_dataset(ds, {"time":1, "z_l":-1})
    
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
    ds = chunk_dataset(ds, {"time":1, "z_l":-1})
    
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
    wm_kwargs = {"coords": coords, "metrics":{}, "padding":{"Z":"extend"}, "autoparse_metadata":False}
    wm_averages = cm4x_watermass(xgcm.Grid(ds[["thetao", "so", "thkcello", "z_i"]], **wm_kwargs))
    ds["sigma2"] = wm_averages.get_density("sigma2")

    _restore_coord_attrs(ds, c_attrs)

    correct_cell_methods(ds)

    ds.attrs["model"] = model
    ds.attrs["description"] = _experiment_description(model)

    return ds

def load_density_annual(odiv, time="*"):
    """Load a CM4X dataset of thermodynamics variables and derive sigma2."""
    state_vars = ["thkcello", "thetao", "so", "rsdo"]
    meta = doralite.dora_metadata(odiv)
    pp = meta['pathPP']
    ppname = "ocean_annual"
    out = "ts"
    local = gu.get_local(pp, ppname, out)
    ds = gu.open_frompp(
        pp, ppname, out, local, time, state_vars,
        dmget=True
    )
    ds = chunk_dataset(ds, {"time":1, "zl":-1, "zi":-1})
    ds = ds.drop_vars(["rsdo"])
    
    c_attrs = {c:ds.coords[c].attrs.copy() for c in ds.coords}
        
    og = gu.open_static(pp, ppname)
    model = [e for e,d in exp_dict.items() for k,v in d.items() if odiv==v][0]
    sg = xr.open_dataset(exp_dict[model]["hgrid"])
    og = fix_geo_coords(og, sg)
    ds = add_grid_coords(ds, og)

    # Compute potential density variables
    coords = {'Z': {'center': 'zl', 'outer': 'zi'}}
    wm_kwargs = {"coords": coords, "metrics":{}, "padding":{"Z":"extend"}, "autoparse_metadata":False}
    wm_averages = cm4x_watermass(xgcm.Grid(ds[["thetao", "so", "thkcello", "zi"]], **wm_kwargs))
    ds["sigma2"] = wm_averages.get_density("sigma2")

    _restore_coord_attrs(ds, c_attrs)

    correct_cell_methods(ds)

    ds.attrs["model"] = model
    ds.attrs["description"] = _experiment_description(model)

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

    grid = ds_to_grid(ds, Zprefix="z_")

    return grid

def regrid_ice(ds, og, ig):
    """Move sea-ice diagnostics onto the ocean tracer grid and merge them in.

    Sea-ice fields arrive on their own grid (`xh_ice`/`yh_ice`). When that grid is
    twice as fine as the ocean tracer grid (the CM4Xp125 case), they are area-weighted
    coarsened by 2x using the ice `CELL_AREA` and ocean `wet` mask; otherwise they are
    simply renamed. The result is merged back onto the ocean `xh`/`yh` coordinates.

    Parameters
    ----------
    ds : `xr.Dataset` containing ocean and (xh_ice/yh_ice) sea-ice diagnostics
    og : ocean static `xr.Dataset` (provides `wet`)
    ig : ice static `xr.Dataset` (provides `CELL_AREA`)

    Returns
    -------
    ds : `xr.Dataset` with sea-ice fields regridded onto the ocean tracer grid
    """
    if "xh_ice" in ds.coords:
        ds_ice = ds.drop_dims(["xh", "yh"])
        ds_ice = ds_ice.rename({"xh_ice":"xh", "yh_ice":"yh"})
        refined = ds.xh_ice.size == 2*ds.xh.size
        if refined:
            ds_ice = ds_ice.assign_coords({
                # Copy-paste fix: this took `og.wet.attrs` (the land mask's), which
                # gave the ice-cell area a mask's `long_name` and, more importantly,
                # `wet`'s mean-mean `cell_methods` instead of an area's sum-sum ones.
                # It only ever fed `grid_ice`'s metric (by value) and is dropped
                # immediately afterwards, so no ice field's numbers depended on it.
                "areacello": xr.DataArray(
                    ig.CELL_AREA.values, dims=("yh", "xh"),
                    attrs={
                        "long_name": "Ocean Grid-Cell Area",
                        "units": "m2",
                        "standard_name": "cell_area",
                        "cell_methods": "area:sum yh:sum xh:sum time:point",
                    },
                ),
                "wet": xr.DataArray(
                    og.wet.values, dims=("yh", "xh"), attrs=og.wet.attrs
                ),
            })
            grid_ice = Grid(
                ds_ice,
                coords={"X":{'center':'xh'},"Y": {'center':'yh'}},
                metrics={('X','Y'): "areacello"},
                padding={"X":"periodic", "Y":"extend"},
                autoparse_metadata=False
            )
            ds_ice = horizontally_coarsen(
                ds_ice,
                grid_ice,
                {"X":2, "Y":2},
                skip_coords=True
            ).drop_vars(["areacello", "wet"])
        for v in ds_ice.data_vars:
            append_provenance(ds_ice[v], (
                "Moved from the sea-ice model grid (`xh_ice`/`yh_ice`) onto the ocean "
                "tracer grid (`xh`/`yh`)"
                + (", which is 2x coarser, by the ice-cell-area-weighted 2x2 mean "
                   "recorded above." if refined else
                   ", which is the same grid; only the dimension names changed.")
            ))
        ds = xr.merge([
            ds.drop_dims(["xh_ice", "yh_ice"]),
            ds_ice.assign_coords({"xh":ds.xh, "yh":ds.yh})
        ])
    return ds
            

def make_wmt_grid(ds, overwrite_grid=True, overwrite_supergrid=True):
    """Make a comprehensive `xwmb`-compatible `xgcm.Grid` object.

    Attaches corrected grid coordinates (from the supergrid, and for CM4Xp125 by
    re-coarsening the full-resolution static file and applying a native-derived 3D
    wet mask), regrids sea ice, derives `boundary_forcing_h_tendency` from `wfo` if
    absent, and computes `sigma2` / `sigma2_bounds` from the (snapshot) state before
    returning the `xgcm.Grid`.

    Parameters
    ----------
    ds : `xr.Dataset` from `load_wmt_ds` (must carry `ds.attrs["model"]`)
    overwrite_grid : if False, keep `ds`'s existing coordinates instead of rebuilding
    overwrite_supergrid : if True, rebuild geo-coordinates from the supergrid

    Returns
    -------
    grid : `xgcm.Grid`
    """

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
                padding={"X":"periodic", "Y":"extend"},
                autoparse_metadata=False
            )
            og = horizontally_coarsen(
                og,
                grid_full,
                {"X":2, "Y":2}
            )

        ds = add_grid_coords(ds, og)

        if ds.attrs["model"] == "CM4Xp125":
            # Compute 3D wet_mask needed to correct budget diagnostics
            pdict_native = get_wmt_pathDict(ds.attrs["model"], "historical", "native", time="1850*")
            ds_native = gu.open_frompp(**pdict_native, dmget=True)
            og_native = gu.open_static(pdict_native["pp"], pdict_native["ppname"])
            ds_native = add_grid_coords(ds_native, og_native)
            
            native_wet_mask = (ds_native['thkcello'].isel(time=0).fillna(0.)!=0.)
            native_wet_mask.attrs["cell_methods"] = "xh:mean yh:mean z_l:sum"
    
            grid_native = Grid(
                ds_native,
                coords={"X":{"center":"xh", "outer":"xq"}, "Y":{"center":"yh", "outer":"yq"}},
                padding={"X":"periodic", "Y":"extend"},
                metrics={('X', 'Y'):['areacello']},
                autoparse_metadata=False
            )

            ds_coarsened_wet_mask = horizontally_coarsen(
                xr.Dataset({'wet_mask': native_wet_mask}).assign_coords(ds_native.coords),
                grid_native,
                dim={"X":2, "Y":2}
            ).fillna(0.)

            # Overwrite d2 coarsened areacello and wet to match my conventions
            ds["areacello"] = ds["areacello"].where(False, ds_coarsened_wet_mask.areacello.values)
            ds["wet"] = ds["wet"].where(False, ds_coarsened_wet_mask.wet.values)
            
            ds_coarsened_wet_mask = ds_coarsened_wet_mask.assign_coords(ds.coords)

            # Parse rather than substring-match: a substring test silently depends
            # on the exact spelling ("xh:mean" vs "xh: mean") of a string this
            # package also rewrites, and a miss here silently skips the wet-mask
            # correction for that variable.
            mms_cell_methods = {"xh": "mean", "yh": "mean", "z_l": "sum"}
            for k,v in ds.data_vars.items():
                if "cell_methods" in v.attrs:
                    cm = parse_cell_methods(v.attrs["cell_methods"])
                    if all([cm.get(d) == m for (d, m) in mms_cell_methods.items()]):
                        ds[k] = xr.DataArray(
                            ds[k] * ds_coarsened_wet_mask.wet_mask,
                            attrs = ds[k].attrs
                        )

        # Add cell_methods to variables if missing (e.g. for sea ice fields)
        for v in [v for v in ds.data_vars if "xh_ice" in ds[v].dims]:
            if "cell_methods" in ds[v].attrs:
                if ds[v].ndim == 4:
                    ds[v].attrs["cell_methods"] = 'area:mean yh:mean xh:mean time: mean'
                else:
                    ds = ds.drop_vars(v)
    
        # Regrid
        print("Regridding ice")
        og = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], "ocean_annual"))
        og = add_grid_coords(og, og)
        og = og.drop_vars(og.data_vars)
        correct_cell_methods(og)
        ig = xr.open_dataset(gu.get_pathstatic(path_dict["pp"], "ice"))
        ds = regrid_ice(ds, og, ig)
    
    grid = ds_to_grid(ds)

    # Correct fsitherm and prlq ocean freshwater flux diagnostics using the RAIN
    # ice diagnostic. The archived ocean `prlq` (liquid precipitation into the
    # ocean) also lumps in the sea-ice melt/freeze (frazil) freshwater flux, which
    # physically belongs to `fsitherm`. We split it back out so that
    #   prlq     -> true rainfall (RAIN, from the ice model), and
    #   fsitherm -> the sea-ice freshwater flux (prlq - RAIN).
    # `fsitherm` is not always archived, and when it is it may be empty, so we only
    # (re)compute it when it is either:
    #   1) missing entirely, or
    #   2) present but empty (all NaN and/or zero) while RAIN differs from prlq.
    # A `fsitherm` that already carries nonzero values is assumed correct and left
    # untouched (case 3).
    if all([e in ds.data_vars for e in ["prlq", "RAIN"]]):
        fsitherm_missing = "fsitherm" not in ds.data_vars
        if fsitherm_missing:
            needs_correction = True
        elif bool((ds["fsitherm"].fillna(0.) == 0.).all()):
            # fsitherm exists but is empty: only worth splitting if RAIN actually
            # differs from prlq (otherwise fsitherm would just be zeros anyway).
            needs_correction = bool(
                (ds["RAIN"].fillna(0.) != ds["prlq"].fillna(0.)).any()
            )
        else:
            # fsitherm already carries nonzero values -> assume it is correct.
            needs_correction = False

        if needs_correction:
            fsitherm = ds["prlq"] - ds["RAIN"]
            if fsitherm_missing:
                # Inherit prlq's metadata (same units and grid staggering) but
                # relabel it as the sea-ice thermodynamic freshwater flux.
                fsitherm.attrs = {
                    **ds["prlq"].attrs,
                    "long_name": (
                        "Water Flux into Sea Water due to Sea Ice Thermodynamics"
                    ),
                    "standard_name": (
                        "water_flux_into_sea_water_due_to_sea_ice_thermodynamics"
                    ),
                }
                ds["fsitherm"] = fsitherm
            else:
                ds["fsitherm"].data = fsitherm.data
            ds["prlq"].data = ds["RAIN"].data
            append_provenance(ds["fsitherm"], (
                "Recomputed by CM4Xutils as `prlq` - `RAIN`: the archived ocean `prlq` "
                "lumps the sea-ice melt/freeze freshwater flux in with liquid "
                "precipitation, so the sea-ice part is split back out here."
            ))
            append_provenance(ds["prlq"], (
                "Replaced by CM4Xutils with the sea-ice model's `RAIN` diagnostic: the "
                "archived ocean `prlq` also contained the sea-ice melt/freeze "
                "freshwater flux, which has been moved to `fsitherm`."
            ))

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
            append_provenance(grid._ds["boundary_forcing_h_tendency"], (
                "Derived by CM4Xutils, not archived by the model: the 2D surface "
                "freshwater flux `wfo` [kg m-2 s-1] was inserted at the ocean surface "
                "and differenced vertically to give a per-layer convergence, then "
                "divided by the Boussinesq reference density RHO_0 = 1035 kg m-3 to "
                "convert it to a thickness tendency [m s-1]. All of it therefore lands "
                "in the topmost layer."
            ))
    
    # Compute potential density variables
    coords = {'Z': grid.axes['Z'].coords}
    wm_kwargs = {"coords": coords, "metrics":{}, "padding":{"Z":"extend"}, "autoparse_metadata":False}
    wm_averages = cm4x_watermass(
        xgcm.Grid(grid._ds[["thetao", "so", "thkcello", coords["Z"]["outer"]]], **wm_kwargs)
    )
    grid._ds["sigma2"] = wm_averages.get_density("sigma2")
    snapshot_state_vars = grid._ds[[
        "thetao_bounds", "so_bounds",
        "thkcello_bounds",
        coords["Z"]["outer"]]
    ]
    rename_vardict = {v:v.split("_")[0] for v in snapshot_state_vars.data_vars}
    wm_snapshots = cm4x_watermass(xgcm.Grid(snapshot_state_vars.rename(rename_vardict), **wm_kwargs))
    grid._ds["sigma2_bounds"] = wm_snapshots.get_density("sigma2")

    _restore_coord_attrs(grid._ds, c_attrs)
    
    return grid

def concat_scenarios(ds_list):
    """Concatenate scenarios in list over all "time" dimensions."""
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
