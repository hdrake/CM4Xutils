"""Descriptive metadata for the generated CM4X data products.

Everything here is *descriptive*: it fills in `units`, `long_name`,
`standard_name`, `comment`, coordinate `axis`/`positive`/`bounds` linkage, and the
dataset-level attributes a reader needs to make sense of a store they have never
seen before. Nothing here changes a single number, and in particular nothing here
writes ``cell_methods`` or ``cell_measures``, which this package *dispatches* on
(see `horizontally_coarsen` and `remap_vertical_coord`) and which therefore live
with the code that consumes them.

`finalize_metadata` is the single entry point; the generation scripts call it
immediately before `to_zarr`.

Two deliberate conventions:

- ``standard_name`` is only ever set where a real CF standard name exists. Most
  MOM6 budget tendencies (`dhdt`, `*_h_tendency`, `T_advection_xy`,
  `Th_tendency_vert_remap`, ...) have no CF equivalent, so they get a `comment`
  instead of an invented name.
- ``units`` strings are only ever *relabeled* into UDUNITS-parseable spellings of
  the same unit (``kg/(m^2*s)`` -> ``kg m-2 s-1``). No value is ever converted; a
  variable whose original spelling was changed records it in ``original_units``.
"""

import datetime
import sys

import numpy as np
import xarray as xr

from .version import __version__

CONVENTIONS = "CF-1.11"

INSTITUTION = (
    "Department of Earth System Science, University of California, Irvine "
    "(postprocessing); NOAA Geophysical Fluid Dynamics Laboratory, Princeton, NJ "
    "(CM4X simulations)"
)

CREATOR_NAME = "Henri F. Drake"
CREATOR_EMAIL = "hfdrake@uci.edu"

REFERENCES = (
    "Griffies et al. (2025), The GFDL-CM4X climate model hierarchy, Part I: "
    "https://doi.org/10.1029/2024MS004861 ; Part II: "
    "https://doi.org/10.1029/2024MS004862 . "
    "Postprocessing code: https://github.com/hdrake/CM4Xutils . "
    "Water mass transformation framework: https://github.com/NOAA-GFDL/xwmt and "
    "https://github.com/hdrake/xwmb ."
)

MODEL_DESCRIPTION = {
    "CM4Xp25": (
        "GFDL CM4X coupled climate model with the MOM6 ocean/sea-ice component on "
        "the nominal 1/4-degree OM4p25 tripolar grid (CM4Xp25)."
    ),
    "CM4Xp125": (
        "GFDL CM4X coupled climate model with the MOM6 ocean/sea-ice component on "
        "the nominal 1/8-degree OM4p125 tripolar grid (CM4Xp125)."
    ),
}

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

# Relabelings only: each maps a model-native spelling onto a UDUNITS-parseable
# spelling of the *same* unit. Nothing here rescales a value.
UNITS_RELABEL = {
    "kg/(m^2*s)": "kg m-2 s-1",
    "kg/(m^2*yr)": "kg m-2 year-1",
    "W/m^2": "W m-2",
    "m-ice": "m",
    "0-1": "1",
    "meters": "m",
    "degrees C": "degC",
}


def normalize_units(ds):
    """Relabel model-native `units` strings into UDUNITS-parseable equivalents.

    Purely a spelling change: the numbers, and the physical unit they are in, are
    untouched. Where a string is changed the original is preserved in
    ``original_units`` so the relabeling is auditable.

    `psu` is deliberately left alone. It is not strictly UDUNITS (CF's canonical
    unit for `sea_water_salinity` is `1e-3`), but it is what the model wrote, it is
    universally understood, and rewriting it would be more confusing than the
    deviation it fixes.
    """
    for v in list(ds.variables):
        attrs = ds[v].attrs
        u = attrs.get("units")
        if u in UNITS_RELABEL:
            attrs["original_units"] = u
            attrs["units"] = UNITS_RELABEL[u]
    return ds


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

# Real CF standard names for variables the model (or this package) left without
# one. Only names verified to exist in the CF standard name table are listed;
# anything else gets a `comment` instead.
STANDARD_NAMES = {
    "siconc": "sea_ice_area_fraction",
    "sithick": "sea_ice_thickness",
    "agessc": "sea_water_age_since_surface_contact",
    "wet": "sea_binary_mask",
    "wet_u": "sea_binary_mask",
    "wet_v": "sea_binary_mask",
    "areacello": "cell_area",
    "deptho": "sea_floor_depth_below_geoid",
}

# Explanations for quantities whose name and long_name do not tell a first-time
# reader what they actually are, or where there is a trap.
COMMENTS = {
    "dhdt": (
        "Total tendency of the layer thickness, i.e. the sum of the dynamics, "
        "boundary-forcing and vertical-remapping thickness tendencies. In sigma2 "
        "coordinates this is the thickness of the density layer, so it includes "
        "water-mass transformation across the layer interfaces."
    ),
    "dynamics_h_tendency": (
        "Layer-thickness tendency due to horizontal (resolved plus parameterized "
        "eddy) transport convergence."
    ),
    "vert_remap_h_tendency": (
        "Layer-thickness tendency due to the model's ALE vertical regridding and "
        "remapping step. In the model's own z*-like coordinate this is a numerical "
        "bookkeeping term; after remapping into density coordinates it is part of "
        "the diathermal/diahaline budget."
    ),
    "boundary_forcing_h_tendency": (
        "Layer-thickness tendency due to surface freshwater forcing. Derived from "
        "`wfo` by CM4Xutils rather than archived by the model (see `provenance`), "
        "so all of it sits in the layer that contains the ocean surface."
    ),
    "Th_tendency_vert_remap": (
        "Heat-content tendency associated with the model's ALE vertical remapping "
        "step. Pairs with `vert_remap_h_tendency`."
    ),
    "Sh_tendency_vert_remap": (
        "Salt-content tendency associated with the model's ALE vertical remapping "
        "step. Pairs with `vert_remap_h_tendency`."
    ),
    "T_advection_xy": (
        "Horizontal convergence of the residual-mean (resolved plus parameterized "
        "eddy) advective heat flux."
    ),
    "S_advection_xy": (
        "Horizontal convergence of the residual-mean (resolved plus parameterized "
        "eddy) advective salt flux."
    ),
    "opottemppmdiff": (
        "This diagnostic is archived as identically zero throughout the CM4X "
        "output, so after the zero-masking applied when coarsening it is entirely "
        "missing (NaN) here. It is retained so the heat budget's term list stays "
        "complete. Note also that MOM6 registers it with `area:sum xh:sum yh:sum` "
        "cell methods despite its `W m-2` (per-area) units; see the repository "
        "history for why that matters."
    ),
    "osaltpmdiff": (
        "This diagnostic is archived as identically zero throughout the CM4X "
        "output, so after the zero-masking applied when coarsening it is entirely "
        "missing (NaN) here. It is retained so the salt budget's term list stays "
        "complete. Note also that MOM6 registers it with `area:sum xh:sum yh:sum` "
        "cell methods despite its `kg m-2 s-1` (per-area) units; see the repository "
        "history for why that matters."
    ),
    "ePBL_h_ML": (
        "Thickness of the surface boundary layer diagnosed by the model's energetic "
        "planetary boundary layer (ePBL) scheme. Scheme-dependent, so it is not "
        "interchangeable with a density- or temperature-threshold mixed layer depth."
    ),
    "heat_content_surfwater": (
        "Heat carried into the ocean by the net surface water flux (liquid plus "
        "frozen), relative to 0 degC. It is the advective heat flux that accompanies "
        "`wfo`, not a turbulent air-sea flux."
    ),
    "vprec": (
        "Virtual liquid precipitation applied by the sea surface salinity restoring "
        "term. A model-configuration flux, not a physical one."
    ),
    "LSNK": (
        "Sea-ice model diagnostic. NOTE THE UNITS: this is a per-YEAR rate, unlike "
        "every other freshwater flux in this dataset, which are per-second. Multiply "
        "by 1/(365*86400) before combining it with them (CM4X uses a NOLEAP calendar)."
    ),
    "LSRC": (
        "Sea-ice model diagnostic. NOTE THE UNITS: this is a per-YEAR rate, unlike "
        "every other freshwater flux in this dataset, which are per-second. Multiply "
        "by 1/(365*86400) before combining it with them (CM4X uses a NOLEAP calendar)."
    ),
    "EVAP": "Sea-ice model evaporation diagnostic, on the ocean tracer grid.",
    "RAIN": "Sea-ice model rainfall diagnostic, on the ocean tracer grid.",
    "SNOWFL": "Sea-ice model snowfall diagnostic, on the ocean tracer grid.",
    "umo": (
        "Mass transport through the eastern face of the tracer cell, integrated over "
        "the face and over the density layer. Positive eastward."
    ),
    "vmo": (
        "Mass transport through the northern face of the tracer cell, integrated over "
        "the face and over the density layer. Positive northward."
    ),
    "thkcello": (
        "Thickness of the density layer, i.e. how much water in this column falls in "
        "this sigma2 bin. Zero/missing where the bin is empty, which is most of the "
        "grid for most bins."
    ),
    "sigma2": (
        "Layer-mean potential density anomaly after remapping. Because the layers are "
        "themselves sigma2 bins, this should sit close to `sigma2_l`; departures "
        "indicate how the density is distributed within the bin."
    ),
    "taux": (
        "Interpolated to tracer cell centers by CM4Xutils; the model applied it at "
        "the u-points (see `provenance`)."
    ),
    "tauy": (
        "Interpolated to tracer cell centers by CM4Xutils; the model applied it at "
        "the v-points (see `provenance`)."
    ),
}


def add_variable_metadata(ds):
    """Fill in missing `units`, `standard_name` and `comment` on data variables.

    Existing attributes always win: this only ever fills gaps, so a value the model
    wrote (or that an earlier pipeline stage deliberately set) is never overwritten.
    """
    for (v, name) in STANDARD_NAMES.items():
        if v in ds.variables:
            ds[v].attrs.setdefault("standard_name", name)

    for (v, comment) in COMMENTS.items():
        if v in ds.variables:
            ds[v].attrs.setdefault("comment", comment)

    # `_bounds` variables are instantaneous snapshots on the `time_bounds` axis, not
    # time means. That is the single most confusable thing about this product -- the
    # suffix looks like a CF cell-bounds variable and is not one -- so say it on every
    # such variable rather than only in the coordinate comment.
    for v in ds.data_vars:
        if str(v).endswith("_bounds"):
            base = str(v)[: -len("_bounds")]
            ds[v].attrs.setdefault("comment", (
                f"Instantaneous snapshot of `{base}` on the `time_bounds` axis, NOT a "
                f"time average and NOT a CF cell-bounds variable. Snapshot i and i+1 "
                f"bracket the monthly mean `{base}` at time i."
            ))

    # Dimensionless masks: give them an explicit unit so a reader does not have to
    # guess whether they are fractions or counts.
    for v in ["wet", "wet_u", "wet_v"]:
        if v in ds.variables:
            ds[v].attrs.setdefault("units", "1")
    if "siconc" in ds.variables:
        ds["siconc"].attrs.setdefault("units", "1")

    # `interp_method: none` is inherited from the archived diagnostic and means "the
    # diag manager did not interpolate this". It is actively false for taux/tauy
    # after CM4Xutils interpolates them to tracer points.
    for v in ["taux", "tauy"]:
        if v in ds.variables:
            ds[v].attrs.pop("interp_method", None)

    return ds


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

COORD_COMMENTS = {
    "xh": (
        "Plain integer index, NOT a longitude. CM4X is on a tripolar grid whose "
        "nominal 1D lon/lat coordinates are meaningless at high latitudes, so they "
        "are replaced by indices here; use the 2D `geolon`/`geolat` for geography."
    ),
    "yh": (
        "Plain integer index, NOT a latitude. Use the 2D `geolon`/`geolat` for "
        "geography."
    ),
    "xq": (
        "Plain integer index of the cell corner/u-face in X, NOT a longitude. "
        "`xq[i]` is the western face of tracer cell `xh[i]`; there is one more `xq` "
        "than `xh`. Use `geolon_u`/`geolat_u` (u-faces) or `geolon_c`/`geolat_c` "
        "(corners) for geography."
    ),
    "yq": (
        "Plain integer index of the cell corner/v-face in Y, NOT a latitude. "
        "`yq[j]` is the southern face of tracer cell `yh[j]`; there is one more `yq` "
        "than `yh`. Use `geolon_v`/`geolat_v` (v-faces) or `geolon_c`/`geolat_c` "
        "(corners) for geography."
    ),
    "exp": (
        "Experiment branch. 'control' is the preindustrial control (piControl, "
        "spinup and continuation splicedtogether); 'forced' is the historical "
        "simulation continued by SSP5-8.5. Both are reported on the same historical "
        "calendar so they can be differenced directly; `time_since_init` recovers "
        "the control's own calendar."
    ),
    "time": (
        "Center of the monthly averaging interval. Both `exp` branches are labeled "
        "on the historical calendar (control years are offset by +1749) so that they "
        "align; `time_since_init` gives the control's original dates."
    ),
    "time_bounds": (
        "Times of the instantaneous snapshots, which bracket the monthly means on "
        "`time`: snapshot `i` and `i+1` are the endpoints of monthly mean `i`, so "
        "there is one more `time_bounds` than `time`. Variables suffixed `_bounds` "
        "live on this axis."
    ),
    "time_since_init": (
        "The control experiment's own model calendar, before it was shifted onto the "
        "historical calendar. Meaningful for exp='control' only."
    ),
    "time_bounds_since_init": (
        "The control experiment's own model calendar for the snapshot times. "
        "Meaningful for exp='control' only."
    ),
    "sigma2_l": (
        "Center of the potential-density layer, referenced to 2000 dbar and minus "
        "1000 kg m-3. These are the model's own 74 archived `rho2` bins plus one "
        "very wide expansion layer at each end, which exist only to catch water "
        "outside the archived range so the conservative remapping cannot spill mass "
        "past the outermost interface. Those two end layers are nominal: their "
        "centers do not correspond to any density the ocean attains."
    ),
    "sigma2_i": (
        "Interfaces bounding the `sigma2_l` layers, one more than there are layers. "
        "The first and last interface are the bracketing values of the two expansion "
        "layers, not physical densities."
    ),
    "rho2_l": (
        "The same coordinate as `sigma2_l` expressed as full potential density "
        "(rho2 = sigma2 + 1000), matching the model's `ocean_month_rho2` diagnostic "
        "axis. Auxiliary: `sigma2_l` is the coordinate this dataset is indexed on."
    ),
    "rho2_i": (
        "The same coordinate as `sigma2_i` expressed as full potential density "
        "(rho2 = sigma2 + 1000). Auxiliary."
    ),
    "lon": "Duplicate of `geolon`, kept for compatibility with `regionate`/`sectionate`.",
    "lat": "Duplicate of `geolat`, kept for compatibility with `regionate`/`sectionate`.",
    "areacello": (
        "Ocean (wet) area of the coarse cell. Multiply an area-mean quantity by this "
        "to recover its cell integral. Divide by `wet` for the total (ocean+land) "
        "cell area."
    ),
    "wet": (
        "Ocean area fraction of the coarse tracer cell: 1 fully ocean, 0 fully land, "
        "intermediate where the coarse cell straddles a coastline."
    ),
    "wet_u": "Ocean fraction of the coarse u-face (eastern/western cell face).",
    "wet_v": "Ocean fraction of the coarse v-face (northern/southern cell face).",
    "dxCv": "Zonal width of the v-face, summed over the wet sub-faces of the coarse face.",
    "dyCu": "Meridional width of the u-face, summed over the wet sub-faces of the coarse face.",
}

# CF axis attributes. The horizontal ones are index coordinates rather than
# longitudes/latitudes, but they are still the X and Y axes of the grid, which is
# what `axis` declares.
COORD_AXES = {
    "xh": "X", "xq": "X", "yh": "Y", "yq": "Y",
    "sigma2_l": "Z", "sigma2_i": "Z",
    "time": "T", "time_bounds": "T",
}


def add_coordinate_metadata(ds):
    """Attach `axis`, `units`, `comment`, and valid `bounds` linkage to coordinates.

    Also repairs two CF violations the pipeline leaves behind:

    - `time` and `time_since_init` inherit ``bounds = "time_bnds"`` from the
      archived diagnostics, but `time_bnds` is dropped by `remap_vertical_coord`
      (it is not on the tracer grid). A dangling `bounds` attribute is invalid CF.
      Where the snapshot axis `time_bounds` is present it brackets the monthly means
      exactly, so a real `time_bnds` is reconstructed from it; otherwise the dangling
      attribute is removed.
    - `sigma2_l` declared its layer edges only through the non-CF ``edges`` attribute
      (an xgcm/MOM6 convention). A CF ``sigma2_bnds`` is added alongside it, built
      from `sigma2_i`. ``edges`` is kept so xgcm-aware code is unaffected.

    The name `sigma2_bnds` is deliberately *not* `sigma2_bounds`, which is already
    taken by the snapshot potential-density field.
    """
    for (c, axis) in COORD_AXES.items():
        if c in ds.coords:
            ds[c].attrs.setdefault("axis", axis)

    # Real CF standard names for the geographic and time coordinates.
    for c in ["geolon", "geolon_u", "geolon_v", "geolon_c", "lon"]:
        if c in ds.coords:
            ds[c].attrs.setdefault("standard_name", "longitude")
    for c in ["geolat", "geolat_u", "geolat_v", "geolat_c", "lat"]:
        if c in ds.coords:
            ds[c].attrs.setdefault("standard_name", "latitude")
    for c in ["time", "time_bounds", "time_since_init", "time_bounds_since_init"]:
        if c in ds.coords:
            ds[c].attrs.setdefault("standard_name", "time")
    if "time_bounds" in ds.coords:
        ds["time_bounds"].attrs.setdefault("long_name", "snapshot time")

    for (c, comment) in COORD_COMMENTS.items():
        if c in ds.coords:
            ds[c].attrs.setdefault("comment", comment)

    for c in ["xh", "yh", "xq", "yq"]:
        if c in ds.coords:
            ds[c].attrs.setdefault("units", "1")

    # `positive` is only meaningful on a dimensional vertical coordinate; on the
    # density axis it is inherited noise, and having it on both `sigma2_*` and the
    # auxiliary `rho2_*` makes the Z axis ambiguous to a CF parser. Keep the axis
    # declaration on the coordinate the dataset is actually indexed on.
    for c in ["rho2_l", "rho2_i"]:
        if c in ds.coords:
            ds[c].attrs.pop("axis", None)

    ds = _add_bounds_variables(ds)
    return ds


def _add_bounds_variables(ds):
    """Build real CF `bounds` variables and drop any dangling `bounds` attribute."""
    # --- vertical layer bounds from the interface coordinate ---
    if ("sigma2_l" in ds.coords) and ("sigma2_i" in ds.coords):
        edges = np.asarray(ds["sigma2_i"].values)
        centers = np.asarray(ds["sigma2_l"].values)
        if edges.size == centers.size + 1:
            ds = ds.assign_coords({
                "sigma2_bnds": xr.DataArray(
                    np.stack([edges[:-1], edges[1:]], axis=-1),
                    dims=("sigma2_l", "nv"),
                    attrs={
                        "long_name": "Bounds of the sigma2 layers",
                        "units": ds["sigma2_l"].attrs.get("units", "kg m-3"),
                        "comment": (
                            "CF bounds for `sigma2_l`, identical information to the "
                            "`sigma2_i` interface coordinate."
                        ),
                    },
                )
            })
            ds["sigma2_l"].attrs["bounds"] = "sigma2_bnds"

    # --- time bounds from the snapshot axis ---
    have_time_bnds = False
    if ("time" in ds.coords) and ("time_bounds" in ds.coords):
        tb = ds["time_bounds"].values
        t = ds["time"].values
        if tb.size == t.size + 1 and np.all(tb[:-1] <= t) and np.all(t <= tb[1:]):
            ds = ds.assign_coords({
                "time_bnds": xr.DataArray(
                    np.stack([tb[:-1], tb[1:]], axis=-1),
                    dims=("time", "nv"),
                    attrs={
                        "long_name": "Bounds of the monthly averaging interval",
                        "comment": (
                            "Reconstructed from the snapshot times on `time_bounds`, "
                            "which are exactly the endpoints of each monthly mean."
                        ),
                    },
                )
            })
            have_time_bnds = True

    for c in ["time", "time_since_init"]:
        if c in ds.coords:
            if have_time_bnds and c == "time":
                ds[c].attrs["bounds"] = "time_bnds"
            else:
                # Dangling: `time_bnds` is not (or no longer) in this dataset.
                ds[c].attrs.pop("bounds", None)

    return ds


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Global attributes inherited from whichever archived NetCDF happened to be merged
# first. They describe that one source file, not this dataset, and several are
# actively wrong here (`title` names only the forced branch; `external_variables`
# points at a `volcello` that has been dropped; `grid_type: regular` is false for a
# tripolar grid; `history`/`NCO`/`filename` are the source file's, not ours).
INHERITED_GLOBAL_ATTRS = [
    "filename", "associated_files", "external_variables",
    "grid_type", "grid_tile", "NCO", "title", "history",
]


def _isoformat(t):
    """ISO-8601 string for a `cftime` or `numpy.datetime64` time value."""
    try:
        return t.isoformat()
    except AttributeError:
        return str(t)


def add_dataset_metadata(
    ds, model=None, product=None, experiment=None, time_range=None,
    version_notes=None, command=None,
):
    """Attach CF/ACDD dataset-level attributes and drop misleading inherited ones.

    Parameters
    ----------
    ds : `xr.Dataset`
    model : "CM4Xp25" or "CM4Xp125"; defaults to ``ds.attrs["model"]``
    product : short product name, e.g. "budgets" or "tracers"
    experiment : experiment name, for products that hold a single branch
    time_range : e.g. "2010-2014"; only used to build the title
    version_notes : prose describing what changed in this dataset release
    command : the command line that generated the file, for `history`
    """
    model = model or ds.attrs.get("model")
    for a in INHERITED_GLOBAL_ATTRS:
        ds.attrs.pop(a, None)

    what = {
        "budgets": "water-mass transformation budget diagnostics",
        "tracers": "transient tracer and ideal-age diagnostics",
    }.get(product, "diagnostics")

    title = f"{model} {what} remapped to sigma2 density coordinates and coarsened"
    if experiment:
        title += f" ({experiment}"
        title += f", {time_range})" if time_range else ")"
    elif time_range:
        title += f" ({time_range})"

    now = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
    command = command or " ".join(sys.argv)
    history = f"{now}: generated by CM4Xutils v{__version__} ({command})"

    attrs = {
        "Conventions": CONVENTIONS,
        "title": title,
        "summary": (
            f"Monthly {what} from the {model} coupled climate simulation, "
            f"conservatively remapped from depth into potential-density (sigma2) "
            f"layers and conservatively coarsened in the horizontal, for water-mass "
            f"transformation analysis with xwmt/xwmb. Each variable's `provenance` "
            f"attribute records exactly which of those steps it went through and how "
            f"it was weighted."
        ),
        "institution": INSTITUTION,
        "source": MODEL_DESCRIPTION.get(model, f"{model} (GFDL CM4X)"),
        "references": REFERENCES,
        "history": history,
        "creator_name": CREATOR_NAME,
        "creator_email": CREATOR_EMAIL,
        "product": product or "",
        "product_version": __version__,
        "source_software": (
            f"CM4Xutils v{__version__} (https://github.com/hdrake/CM4Xutils)"
        ),
        "comment": (
            "Non-standard conventions a reader should know about. (1) `cell_methods` "
            "uses the compact `dim:method` spelling (no space after the colon) rather "
            "than the CF `dim: method` spelling, because CM4Xutils parses and "
            "dispatches on these strings; the meaning is the CF meaning. (2) The "
            "horizontal dimension coordinates `xh`/`yh`/`xq`/`yq` are integer indices, "
            "not longitudes and latitudes -- use the 2D `geolon*`/`geolat*` "
            "coordinates. (3) Zero-valued cells are written as missing (NaN) by the "
            "coarsening step, so a NaN means either land, an empty density layer, or "
            "a genuine zero."
        ),
    }
    if experiment:
        attrs["experiment"] = experiment
    if version_notes:
        attrs["version_notes"] = version_notes

    for c in ["time_bounds", "time"]:
        if c in ds.coords and ds.sizes.get(c, 0):
            vals = np.asarray(ds[c].values)
            attrs["time_coverage_start"] = _isoformat(vals[0])
            attrs["time_coverage_end"] = _isoformat(vals[-1])
            attrs["calendar"] = getattr(vals[0], "calendar", "") or ""
            break

    # Never clobber `model`, `description` or the narrative `provenance` built up by
    # the pipeline; they are the dataset's own, not inherited noise.
    ds.attrs.update(attrs)
    if model is not None:
        ds.attrs["model"] = model
    return ds


def finalize_metadata(ds, **kwargs):
    """Apply all descriptive metadata to a finished data product.

    Call this last, immediately before writing: earlier pipeline stages reassign
    ``.attrs`` wholesale in several places and would drop anything set before them.

    ``kwargs`` are forwarded to `add_dataset_metadata`.
    """
    ds = normalize_units(ds)
    ds = add_variable_metadata(ds)
    ds = add_coordinate_metadata(ds)
    ds = add_dataset_metadata(ds, **kwargs)
    return ds
