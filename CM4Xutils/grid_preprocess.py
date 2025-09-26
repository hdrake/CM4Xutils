import os
import xarray as xr
import numpy as np
from xgcm import Grid

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
        raise ValueError("ocean grid must be symmetric")
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
        'xh': xr.DataArray(np.arange(og.xh.size), dims=("xh",), attrs=og.xh.attrs),
        'yh': xr.DataArray(np.arange(og.yh.size), dims=("yh",), attrs=og.yh.attrs),
        'xq': xr.DataArray(np.arange(og.xq.size), dims=("xq",), attrs=og.xq.attrs),
        'yq': xr.DataArray(np.arange(og.yq.size), dims=("yq",), attrs=og.yq.attrs),
    })
    if ("thkcello" not in ds) and ("volcello" in ds) and ("areacello" in ds):
        ds['thkcello'] = ds['volcello']/ds['areacello']

    correct_cell_methods(ds)

    return ds

def ds_to_grid(ds, Zprefix=None):
    """Instantiate a `xwmb`-compatiable `xgcm.Grid` object.
    
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
    
    boundary = {"X":"periodic", "Y":"extend", "Z":"extend"}
    
    return Grid(
        ds,
        coords=coords,
        metrics=metrics,
        boundary=boundary,
        autoparse_metadata=False
    )

def add_sigma2_coords(ds):
    """Add the standard CM4X 72-layer sigma2 coordinates to dataset.

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
            "description": "Computed offline using the gsw python package implementation of TEOS10.",
        }
    if "sigma2_bounds" in ds.data_vars:
        ds.sigma2_bounds.attrs = {
            "long_name": "Potential Density referenced to 2000 dbar (minus 1000 kg/m3)",
            "units": "kg m-3",
            "cell_methods": "area:mean z_l:mean yh:mean xh:mean time:point",
            "volume": "volcello",
            "area": "areacello",
            "description": "Computed offline using the gsw python package implementation of TEOS10."
        }

    return ds

def correct_cell_methods(ds):
    """Correct cell methods for depth and wet mask coordinates.

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
    split_list = replace_by_dict(s, {" : ":":", ": ":":", " :":":"}).split(" ")
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
