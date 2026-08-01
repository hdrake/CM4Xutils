"""Grid coordinate fixes and `xgcm.Grid` construction.

This module has no GFDL-specific dependencies and is imported by every other
module in the package. It provides the building blocks of the loading pipeline:

- `fix_geo_coords` / `add_grid_coords` reconstruct the (wrong) static-file geo
  coordinates from the authoritative supergrid and attach them to a dataset.
- `ds_to_grid` wraps a prepared dataset in an `xwmb`-compatible `xgcm.Grid`.
- `add_sigma2_coords` attaches the target density coordinate used for remapping.
- The `*_cell_methods` helpers parse and serialize the ``cell_methods`` attribute
  strings that both coarsening and vertical remapping dispatch on.
"""

import os
import xarray as xr
import numpy as np
from xgcm import Grid

# Bracketing interfaces of the expanded sigma2 grid built by `add_sigma2_coords`
# [kg m-3]. The archived CM4X coordinate spans sigma2 = [-3, 39]; one interface is
# added below and above that range so the conservative remapping has somewhere to
# put water outside it instead of spilling mass past the outermost layers. Kept
# just wide enough to cover any plausible ocean density -- widening them only
# stretches the two expansion layers and pushes their nominal centers away from
# any density the ocean actually attains.
SIGMA2_MIN = -10.
SIGMA2_MAX = 50.

def fix_geo_coords(og, sg):
    """Fix geographical coordinates from static file with supergrid

    The geographical coordinates (e.g. `geolon` and geolat`) in CM4X
    static files are generally wrong. The true CM4X grid information
    resides in the supergrid ("hgrid") file, which contains the
    coordinates of all horizontal cell centers, faces, and corners.

    This function additionally infers whether the static file has
    already been coarsened by a factor of 2 (as for "d2" diagnostics)
    and also corrects those coordinates from the supergrid.

    Parameters
    ----------
    og : `xr.Dataset` containing CM4X static file grid coordinates
    sg : `xr.Dataset` containing CM4X supergrid (or "hgrid") coordinates

    Returns
    -------
    og : A corrected `xr.Dataset` containing CM4X grid coordinates
    
    """
    # The supergrid ("hgrid") is a doubly-refined mesh: it stores cell centers,
    # faces, and corners on a single array at twice the tracer-grid resolution.
    # We recover each staggered location by strided slicing:
    #   - centers (h,h) at odd indices, corners (q,q) at even indices,
    #   - u-faces at (odd y, even x), v-faces at (even y, odd x).
    # A native static file has xh == nx//2; a "d2" (2x-coarsened) file has
    # xh == nx//4, in which case the strides double from 2 to 4.
    if og.sizes['xh'] == sg.sizes['nx']//2:
        og = og.assign_coords({
            'geolon'  : xr.DataArray(sg['x'][1::2,1::2].data, dims=("yh", "xh"), attrs=og.geolon.attrs),
            'geolat'  : xr.DataArray(sg['y'][1::2,1::2].data, dims=("yh", "xh"), attrs=og.geolat.attrs),
            'geolon_u': xr.DataArray(sg['x'][1::2,0::2].data, dims=("yh", "xq"), attrs=og.geolon_u.attrs),
            'geolat_u': xr.DataArray(sg['y'][1::2,0::2].data, dims=("yh", "xq"), attrs=og.geolat_u.attrs),
            'geolon_v': xr.DataArray(sg['x'][0::2,1::2].data, dims=("yq", "xh"), attrs=og.geolon_v.attrs),
            'geolat_v': xr.DataArray(sg['y'][0::2,1::2].data, dims=("yq", "xh"), attrs=og.geolat_v.attrs),
            'geolon_c': xr.DataArray(sg['x'][0::2,0::2].data, dims=("yq", "xq"), attrs=og.geolon_c.attrs),
            'geolat_c': xr.DataArray(sg['y'][0::2,0::2].data, dims=("yq", "xq"), attrs=og.geolat_c.attrs)
        })
    elif og.sizes['xh'] == sg.sizes['nx']//4:
        og = og.assign_coords({
            'geolon'  : xr.DataArray(sg['x'][2::4,2::4].data, dims=("yh", "xh"), attrs=og.geolon.attrs),
            'geolat'  : xr.DataArray(sg['y'][2::4,2::4].data, dims=("yh", "xh"), attrs=og.geolat.attrs),
            'geolon_u': xr.DataArray(sg['x'][2::4,0::4].data, dims=("yh", "xq"), attrs=og.geolon_u.attrs),
            'geolat_u': xr.DataArray(sg['y'][2::4,0::4].data, dims=("yh", "xq"), attrs=og.geolat_u.attrs),
            'geolon_v': xr.DataArray(sg['x'][0::4,2::4].data, dims=("yq", "xh"), attrs=og.geolon_v.attrs),
            'geolat_v': xr.DataArray(sg['y'][0::4,2::4].data, dims=("yq", "xh"), attrs=og.geolat_v.attrs),
            'geolon_c': xr.DataArray(sg['x'][0::4,0::4].data, dims=("yq", "xq"), attrs=og.geolon_c.attrs),
            'geolat_c': xr.DataArray(sg['y'][0::4,0::4].data, dims=("yq", "xq"), attrs=og.geolat_c.attrs)
        })
    else:
        raise ValueError(
            f"Could not match ocean grid (xh={og.sizes['xh']}) to the supergrid "
            f"(nx={sg.sizes['nx']}): expected xh to equal nx//2 (native) or "
            f"nx//4 (d2-coarsened). Check that `og` and `sg` are the same model."
        )
    return og
    
def add_grid_coords(ds, og):
    """Assigns ocean grid coordinates to a dataset with diagnostic variables.

    Parameters
    ----------
    ds : `xr.Dataset` containing CM4X diagnostics
    og : `xr.Dataset` containing CM4X coordinates
        Must contain the following coordinates, which are standard in
        static files: ["areacello", "geolon", "geolat", "geolon_c",
        "geolat_c", "geolon_u", "geolat_u", "geolon_v", "geolat_v",
        "deptho", "areacello", "wet", "wet_u", "wet_v"].

        Should ideally contain ["dxCv", "dyCu"] as well, but optional.

    Returns
    -------
    ds : `xr.Dataset` containing both CM4X diagnostics and coordinates
    """
    
    og['deptho'] = (
        og['deptho'].where(~np.isnan(og['deptho']), 0.)
    )

    if all([c in og for c in ["dxCv", "dyCu"]]):
        # add velocity face widths to calculate distances along the section
        ds = ds.assign_coords({
            'dxCv': xr.DataArray(
                og['dxCv'].transpose('xh', 'yq').values, dims=('xh', 'yq',),
                attrs={**og.dxCv.attrs, **{"cell_methods": "xh:sum yq:point time:point"}},
            ),
            'dyCu': xr.DataArray(
                og['dyCu'].transpose('xq', 'yh').values, dims=('xq', 'yh',),
                attrs={**og.dyCu.attrs, **{"cell_methods": "xq:point yh:sum time:point"}},
            )
        })
    
    ds = ds.assign_coords({
        'areacello':xr.DataArray(og['areacello'].values, dims=("yh", "xh"), attrs=og.areacello.attrs),
        'geolon':   xr.DataArray(og['geolon'].values, dims=("yh", "xh"), attrs=og.geolon.attrs),
        'lon':      xr.DataArray(og['geolon'].values, dims=("yh", "xh"), attrs=og.geolon.attrs),
        'geolat':   xr.DataArray(og['geolat'].values, dims=("yh", "xh"), attrs=og.geolat.attrs),
        'lat':      xr.DataArray(og['geolat'].values, dims=("yh", "xh"), attrs=og.geolat.attrs),
        'geolon_u': xr.DataArray(og['geolon_u'].values, dims=("yh", "xq",), attrs=og.geolon_u.attrs),
        'geolat_u': xr.DataArray(og['geolat_u'].values, dims=("yh", "xq",), attrs=og.geolat_u.attrs),
        'geolon_v': xr.DataArray(og['geolon_v'].values, dims=("yq", "xh",), attrs=og.geolon_v.attrs),
        'geolat_v': xr.DataArray(og['geolat_v'].values, dims=("yq", "xh",), attrs=og.geolat_v.attrs),
        'geolon_c': xr.DataArray(og['geolon_c'].values, dims=("yq", "xq",), attrs=og.geolon_c.attrs),
        'geolat_c': xr.DataArray(og['geolat_c'].values, dims=("yq", "xq",), attrs=og.geolat_c.attrs),
        'deptho':   xr.DataArray(og['deptho'].values, dims=("yh", "xh",), attrs=og.deptho.attrs),
        'wet':   xr.DataArray(og['wet'].values,   dims=("yh", "xh",), attrs=og.wet.attrs),
        'wet_u': xr.DataArray(og['wet_u'].values, dims=("yh", "xq",), attrs=og.wet_u.attrs),
        'wet_v': xr.DataArray(og['wet_v'].values, dims=("yq", "xh",), attrs=og.wet_v.attrs),
        # Replace the (unreliable) nominal lat/lon dimension coordinates with plain
        # integer indices; downstream code navigates in 2D via geolat/geolon instead.
        'xh': xr.DataArray(np.arange(og.xh.size), dims=("xh",), attrs=og.xh.attrs),
        'yh': xr.DataArray(np.arange(og.yh.size), dims=("yh",), attrs=og.yh.attrs),
        'xq': xr.DataArray(np.arange(og.xq.size), dims=("xq",), attrs=og.xq.attrs),
        'yq': xr.DataArray(np.arange(og.yq.size), dims=("yq",), attrs=og.yq.attrs),
    })
    # Derive cell thickness from cell volume when it is not archived directly,
    # so that volume-weighting is available downstream (added in v1.3.0).
    if ("thkcello" not in ds) and ("volcello" in ds) and ("areacello" in ds):
        ds['thkcello'] = ds['volcello']/ds['areacello']
        ds['thkcello'].attrs = {
            'long_name': 'Cell Thickness',
            'units': 'm',
            'cell_methods': 'area:mean z_l:sum yh:mean xh:mean time: mean',
            'cell_measures': 'volume: volcello area: areacello',
            'time_avg_info': 'average_T1,average_T2,average_DT',
            'standard_name': 'cell_thickness'
        }
        
    correct_cell_methods(ds)

    return ds

def ds_to_grid(ds, Zprefix=None):
    """Instantiate a `xwmb`-compatible `xgcm.Grid` object.
    
    Parameters
    ----------
    ds : `xr.Dataset` containing CM4X data variables and coordinates
    Zprefix : `str` describing the dataset's vertical coordinate (default: `None`)
        If `None`, then it attempts to infer the vertical coordinate from the
        names of the dataset's dimensions.
        Support options: ["sigma2", "rho2", "z", "z_"]

    Returns
    -------
    grid : `xgcm.Grid` object formatted as required by the `sectionate` and `regionate`
    packages as well as the `xwmt.WaterMassTransformation` and `xwmb.WaterMassBudget`
    constructor methods.
    """
    coords={
        'X': {k:v for (k,v) in {'center':'xh','outer':'xq'}.items()
              if v in ds},
        'Y': {k:v for (k,v) in {'center':'yh','outer':'yq'}.items()
              if v in ds},
    }
    if Zprefix is not None:
        if "z" in Zprefix:
            coords = {
                **coords,
                **{'Z': {'center': f'{Zprefix}l', 'outer': f'{Zprefix}i'}}
            }
    else:
        print("Inferring Z grid coordinate: ", end="")
        if "sigma2_l" in ds.dims:
            coords = {
                **coords,
                **{'Z': {'center': 'sigma2_l', 'outer': 'sigma2_i'}}
            }
            print("density `sigma2`")
        elif "rho2_l" in ds.dims:
            coords = {
                **coords,
                **{'Z': {'center': 'rho2_l', 'outer': 'rho2_i'}}
            }
            print("density `rho2`")
        elif "zl" in ds.dims:
            coords = {
                **coords,
                **{'Z': {'center': 'zl', 'outer': 'zi'}}
            }
            print("native `z`")
        elif "z_l" in ds.dims:
            coords = {
                **coords,
                **{'Z': {'center': 'z_l', 'outer': 'z_i'}}
            }
            print("depth `z_`")
        
    if "areacello" in ds:
        metrics = {
            ('X','Y'): "areacello",
        }
    else:
        metrics = {}
    
    padding = {"X":"periodic", "Y":"extend", "Z":"extend"}

    return Grid(
        ds,
        coords=coords,
        metrics=metrics,
        padding=padding,
        autoparse_metadata=False
    )

def add_sigma2_coords(ds):
    """Add the standard CM4X 74-layer sigma2 coordinates to dataset.

    Parameters
    ----------
    ds : `xr.Dataset`

    Returns
    -------
    ds : `xr.Dataset` containing target sigma2 coordinates
    """
    if not(all(c in ds.coords for c in ["sigma2_l", "sigma2_i"])):
        # Set up target coordinates
        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "../data/sigma2_coords.nc")
        sigma2_coords = xr.open_dataset(filename)
        # Pad one extra interface at each end (at SIGMA2_MIN / SIGMA2_MAX) so that
        # the target grid brackets every plausible ocean density and the
        # conservative remapping never spills mass past the outermost layers.
        sigma2_coords_expanded = xr.Dataset(
            coords={
                "sigma2_i": xr.DataArray(
                    np.concatenate((
                        [SIGMA2_MIN],
                        sigma2_coords.sigma2_i.values,
                        [SIGMA2_MAX]
                    )),
                    dims=("sigma2_i",),
                    attrs=sigma2_coords.sigma2_i.attrs
                ),
                "rho2_i": xr.DataArray(
                    np.concatenate((
                        [SIGMA2_MIN + 1000.],
                        sigma2_coords.rho2_i.values,
                        [SIGMA2_MAX + 1000.]
                    )),
                    dims=("sigma2_i",),
                    attrs=sigma2_coords.rho2_i.attrs
                ),
            },
            attrs={"description":"CM4X 74-layer sigma2 coordinate grid, expanded on both ends to include all plausible ocean densities."}
        )
        sigma2_coords_expanded = sigma2_coords_expanded.assign_coords({
            "sigma2_l": xr.DataArray(
                np.concatenate((
                    [np.mean(sigma2_coords_expanded.sigma2_i.values[0:2])],
                    sigma2_coords.sigma2_l.values,
                    [np.mean(sigma2_coords_expanded.sigma2_i.values[-2:])]
                )),
                dims=("sigma2_l",),
                attrs=sigma2_coords.sigma2_l.attrs
            ),
            "rho2_l": xr.DataArray(
                np.concatenate((
                    [np.mean(sigma2_coords_expanded.rho2_i.values[0:2])],
                    sigma2_coords.rho2_l.values,
                    [np.mean(sigma2_coords_expanded.rho2_i.values[-2:])]
                )),
                dims=("sigma2_l",),
                attrs=sigma2_coords.rho2_l.attrs
            ),
        })
        sigma2_coords = sigma2_coords_expanded

        for c in sigma2_coords.dims:
            sigma2_coords.coords[c].attrs = sigma2_coords.coords[c.replace("sigma2", "rho2")].attrs
            sigma2_coords.coords[c].attrs["long_name"] = sigma2_coords.coords[c].attrs["long_name"].replace(
                "Potential Density", "Potential Density minus 1000 kg/m3"
            )
            sigma2_coords.coords[c].attrs["cell_methods"] = f"{c}:point"
        sigma2_coords.coords["sigma2_l"].attrs["edges"] = "sigma2_i"
    
        # Drop unnecessary or redundant variables
        drop_vars = [
            "obvfsq", "rsdo", "volcello", "volcello_bounds",
            "uo", "vo", "uhml", "vhml"
        ]
        ds = xr.merge([
            ds.drop_vars([v for v in drop_vars if v in ds]),
            sigma2_coords
        ])

    # Add attributes for sigma2
    if "sigma2" in ds.data_vars:
        ds.sigma2.attrs = {
            "long_name": "Potential Density referenced to 2000 dbar (minus 1000 kg/m3)",
            "units": "kg m-3",
            "cell_methods": "area:mean z_l:mean yh:mean xh:mean time:mean",
            "volume": "volcello",
            "area": "areacello",
            "time_avg_info": "average_T1,average_T2,average_DT",
            "equation_of_state": "wright97-reduced (xeos; MOM6 EQN_OF_STATE=WRIGHT)",
            "description": (
                "Computed offline with the MOM6 Wright (1997) reduced-range equation of "
                "state via xeos (wright97-reduced), matching the CM4X model configuration "
                "EQN_OF_STATE='WRIGHT' to machine precision (identical coefficients and "
                "density formula; differs from MOM6's legacy kernel only by floating-point "
                "addition associativity, ~1e-12 kg/m3)."
            ),
        }
    if "sigma2_bounds" in ds.data_vars:
        ds.sigma2_bounds.attrs = {
            "long_name": "Potential Density referenced to 2000 dbar (minus 1000 kg/m3)",
            "units": "kg m-3",
            "cell_methods": "area:mean z_l:mean yh:mean xh:mean time:point",
            "volume": "volcello",
            "area": "areacello",
            "equation_of_state": "wright97-reduced (xeos; MOM6 EQN_OF_STATE=WRIGHT)",
            "description": (
                "Computed offline with the MOM6 Wright (1997) reduced-range equation of "
                "state via xeos (wright97-reduced), matching the CM4X model configuration "
                "EQN_OF_STATE='WRIGHT' to machine precision (identical coefficients and "
                "density formula; differs from MOM6's legacy kernel only by floating-point "
                "addition associativity, ~1e-12 kg/m3)."
            ),
        }

    return ds

def correct_cell_methods(ds):
    """Correct cell methods for depth and wet mask coordinates.

    These static-file coordinates are missing (or carry incorrect) ``cell_methods``
    attributes, so their coarsening behavior would otherwise be undefined. This
    stamps the correct methods so `horizontally_coarsen` treats each one properly:
    `wet`/`deptho` as tracer-cell means, `wet_u`/`wet_v` as face quantities.

    Modifies `ds` in place (does not return a new dataset).

    Parameters
    ----------
    ds : `xr.Dataset`
    """
    def correct_cell_method(v, cell_methods):
        if v in list(ds.data_vars)+list(ds.coords):
            ds[v].attrs["cell_methods"] = cell_methods
        
    correct_cell_method("wet", "xh:mean yh:mean time:point")
    correct_cell_method("wet_u", "xq:point yh:mean time:point")
    correct_cell_method("wet_v", "xh:mean yq:point time:point")
    correct_cell_method("deptho", "xh:mean yh:mean time:point")

def replace_by_dict(s, d):
    """Apply multiple string replacements by looping through a dictionary"""
    for k,v in d.items():
        s = s.replace(k,v)
    return s

def parse_cell_methods(s):
    """Parse cell method string as dictionary

    Parameters
    ----------
    s : cell method str
        Must be formatted as a single string with dimensions
        and their cell methods separated by `":"` and each pair
        separated by a space `" "`.
        Example: `"xh:mean yh:mean time:point"`

    Returns
    -------
    d : dictionary mapping dimensions to their cell methods
        Example: `{"xh":"mean", "yh":"mean", "time":"point"}`
    """
    # Normalize whitespace around the ":" separators before splitting, so that
    # inconsistently formatted strings (e.g. "time: mean") parse the same way.
    split_list = replace_by_dict(s, {" : ":":", ": ":":", " :":":"}).split(" ")
    for e in split_list:
        if ":" not in e:
            raise ValueError(
                f"Malformed cell_methods string {s!r}: token {e!r} is missing a "
                f"'dim:method' pair separated by ':'."
            )
    d = {e.split(":")[0]:e.split(":")[1] for e in split_list}
    return d

def stringify_cell_methods_dict(d):
    """Turn cell method dictionary into str
        Parameters
    ----------
    d : dictionary mapping dimensions to their cell methods
        Example: `{"xh":"mean", "yh":"mean", "time":"point"}`

    Returns
    -------
    s : cell method str
        Must be formatted as a single string with dimensions
        and their cell methods separated by `":"` and each pair
        separated by a space `" "`.
        Example: `"xh:mean yh:mean time:point"`
    """
    s = replace_by_dict(str(d), {"'":"", ",":"", ": ":":", "{":"", "}":""})
    return s
