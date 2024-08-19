import xarray as xr
import numpy as np
from xgcm import Grid

def fix_geo_coords(og, sg):
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
    og['deptho'] = (
        og['deptho'].where(~np.isnan(og['deptho']), 0.)
    )
    
    ds = ds.assign_coords({
        'dxCv': xr.DataArray(
            og['dxCv'].transpose('xh', 'yq').values, dims=('xh', 'yq',),
            attrs={**og.dxCv.attrs, **{"cell_methods": "xh:sum yq:point time:point"}},
        ),
        'dyCu': xr.DataArray(
            og['dyCu'].transpose('xq', 'yh').values, dims=('xq', 'yh',),
            attrs={**og.dyCu.attrs, **{"cell_methods": "xq:point yh:sum time:point"}},
        )
    }) # add velocity face widths to calculate distances along the section
    
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
        'wet_v':     xr.DataArray(og['wet_v'].values, dims=("yq", "xh",), attrs=og.wet_v.attrs),
        'wet_u':     xr.DataArray(og['wet_u'].values, dims=("yh", "xq",), attrs=og.wet_u.attrs),
        'wet':     xr.DataArray(og['wet'].values, dims=("yh", "xh",), attrs=og.wet.attrs),
    })
    if ("thkcello" not in ds) and ("volcello" in ds) and ("areacello" in ds):
        ds['thkcello'] = ds['volcello']/ds['areacello']

    correct_cell_methods(ds)

    return ds

def ds_to_grid(ds):
    
    coords={
        'X': {k:v for (k,v) in {'center':'xh','outer':'xq'}.items()
              if v in ds},
        'Y': {k:v for (k,v) in {'center':'yh','outer':'yq'}.items()
              if v in ds},
    }
    if "sigma2_l" in ds.dims:
        coords = {
            **coords,
            **{'Z': {'center': 'sigma2_l', 'outer': 'sigma2_i'}}
        }
    elif "rho2_l" in ds.dims:
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

def correct_cell_methods(ds):
    def correct_cell_method(v, cell_methods):
        ds[v].attrs["cell_methods"] = cell_methods
        
    correct_cell_method("wet", "xh:mean yh:mean time:point")
    correct_cell_method("wet_u", "xq:mean yh:point time:point")
    correct_cell_method("wet_v", "xh:point yq:mean time:point")
    correct_cell_method("deptho", "xh:mean yh:mean time:point")

def replace_by_dict(s, d):
    for k,v in d.items():
        s = s.replace(k,v)
    return s

def parse_cell_methods(s):
    split_list = replace_by_dict(s, {" : ":":", ": ":":", " :":":"}).split(" ")
    return {e.split(":")[0]:e.split(":")[1] for e in split_list}

def stringify_cell_methods_dict(d):
    return replace_by_dict(str(d), {"'":"", ",":"", ": ":":", "{":"", "}":""})