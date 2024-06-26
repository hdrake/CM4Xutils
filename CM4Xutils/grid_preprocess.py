import xarray as xr
import numpy as np
from xgcm import Grid

def fix_geo_coords(og, sg):
    if og.sizes['xh'] == sg.sizes['nx']//2:
        og = og.assign_coords({
            'geolon'  : xr.DataArray(sg['x'][1::2,1::2].data, dims=["yh", "xh"]),
            'geolat'  : xr.DataArray(sg['y'][1::2,1::2].data, dims=["yh", "xh"]),
            'geolon_u': xr.DataArray(sg['x'][1::2,0::2].data, dims=["yh", "xq"]),
            'geolat_u': xr.DataArray(sg['y'][1::2,0::2].data, dims=["yh", "xq"]),
            'geolon_v': xr.DataArray(sg['x'][0::2,1::2].data, dims=["yq", "xh"]),
            'geolat_v': xr.DataArray(sg['y'][0::2,1::2].data, dims=["yq", "xh"]),
            'geolon_c': xr.DataArray(sg['x'][0::2,0::2].data, dims=["yq", "xq"]),
            'geolat_c': xr.DataArray(sg['y'][0::2,0::2].data, dims=["yq", "xq"])
        })
    elif og.sizes['xh'] == sg.sizes['nx']//4:
        og = og.assign_coords({
            'geolon'  : xr.DataArray(sg['x'][2::4,2::4].data, dims=["yh", "xh"]),
            'geolat'  : xr.DataArray(sg['y'][2::4,2::4].data, dims=["yh", "xh"]),
            'geolon_u': xr.DataArray(sg['x'][2::4,0::4].data, dims=["yh", "xq"]),
            'geolat_u': xr.DataArray(sg['y'][2::4,0::4].data, dims=["yh", "xq"]),
            'geolon_v': xr.DataArray(sg['x'][0::4,2::4].data, dims=["yq", "xh"]),
            'geolat_v': xr.DataArray(sg['y'][0::4,2::4].data, dims=["yq", "xh"]),
            'geolon_c': xr.DataArray(sg['x'][0::4,0::4].data, dims=["yq", "xq"]),
            'geolat_c': xr.DataArray(sg['y'][0::4,0::4].data, dims=["yq", "xq"])
        })
    else:
        raise ValueError("ocean grid must be symmetric")
    return og
    
def add_grid_coords(ds, og):
    og['deptho'] = (
        og['deptho'].where(~np.isnan(og['deptho']), 0.)
    )
    
    ds = ds.assign_coords({
        'dxCv': xr.DataArray(
            og['dxCv'].transpose('xh', 'yq').values, dims=('xh', 'yq',)
        ),
        'dyCu': xr.DataArray(
            og['dyCu'].transpose('xq', 'yh').values, dims=('xq', 'yh',)
        )
    }) # add velocity face widths to calculate distances along the section
    
    ds = ds.assign_coords({
        'areacello':xr.DataArray(og['areacello'].values, dims=("yh", "xh")),
        'geolon':   xr.DataArray(og['geolon'].values, dims=("yh", "xh")),
        'lon':      xr.DataArray(og['geolon'].values, dims=("yh", "xh")),
        'geolat':   xr.DataArray(og['geolat'].values, dims=("yh", "xh")),
        'lat':      xr.DataArray(og['geolat'].values, dims=("yh", "xh")),
        'geolon_u': xr.DataArray(og['geolon_u'].values, dims=("yh", "xq",)),
        'geolat_u': xr.DataArray(og['geolat_u'].values, dims=("yh", "xq",)),
        'geolon_v': xr.DataArray(og['geolon_v'].values, dims=("yq", "xh",)),
        'geolat_v': xr.DataArray(og['geolat_v'].values, dims=("yq", "xh",)),
        'geolon_c': xr.DataArray(og['geolon_c'].values, dims=("yq", "xq",)),
        'geolat_c': xr.DataArray(og['geolat_c'].values, dims=("yq", "xq",)),
        'deptho':   xr.DataArray(og['deptho'].values, dims=("yh", "xh",)),
        'wet_v':     xr.DataArray(og['wet_v'].values, dims=("yq", "xh",)),
        'wet_u':     xr.DataArray(og['wet_u'].values, dims=("yh", "xq",)),
        'wet':     xr.DataArray(og['wet'].values, dims=("yh", "xh",)),
    })
    if ("thkcello" not in ds) and ("volcello" in ds) and ("areacello" in ds):
        ds['thkcello'] = ds['volcello']/ds['areacello']
    
    return ds

def ds_to_grid(ds):
    
    coords={
        'X': {k:v for (k,v) in {'center':'xh','outer':'xq'}.items()
              if v in ds},
        'Y': {k:v for (k,v) in {'center':'yh','outer':'yq'}.items()
              if v in ds},
    }
    if "rho2_l" in ds.dims:
        coords = {
            **coords,
            **{'Z': {'center': 'rho2_l', 'outer': 'rho2_i'}}
        }
    elif "zl" in ds.dims:
        coords = {
            **coords,
            **{'Z': {'center': 'zl', 'outer': 'zi'}}
        }
    elif "z_l" in ds.dims:
        coords = {
            **coords,
            **{'Z': {'center': 'z_l', 'outer': 'z_i'}}
        }
        
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