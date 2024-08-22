from .grid_preprocess import *

def horizontally_coarsen(ds, grid, dim):
    ds_coarse = xr.Dataset(attrs=ds.attrs)

    coord_vars = [e for e in ["areacello", "deptho", "wet", "wet_u", "wet_v"] if e in ds.coords]
    for v in list(ds.data_vars) + coord_vars:
        da = ds[v]

        if "cell_methods" not in da.attrs:
            print(f"Skipping variable {v} because `cell_methods` attribute not defined.")
            continue
            
        cell_method = {k:v for (k,v) in parse_cell_methods(da.cell_methods).items()}
        try:
            dim_names = {
                "X": grid._get_dims_from_axis(da, "X")[0],
                "Y": grid._get_dims_from_axis(da, "Y")[0]
            }
        except:
            print(f"Skipping {v} because independent of 'X' and 'Y' dims.")
            continue

        for (dim_name, dim_var) in dim_names.items():
            if cell_method[dim_var] == "sum":
                da = da.coarsen(dim={dim_var:dim[dim_name]}).sum()
                da = da.where(da!=0.)
            elif cell_method[dim_var] == "point":
                da = da.sel({dim_var: da[dim_var][::dim[dim_name]]})

        if all([cell_method[dim_var] == "mean" for dim_var in dim_names.values()]):
            A = grid.get_metric(da, ["X", "Y"])*ds.wet
            attrs = da.attrs
            Zcenter = grid.axes["Z"].coords["center"]
            cdim = {dim_var:dim[dim_name] for (dim_name, dim_var) in dim_names.items()}
            if Zcenter not in cell_method:
                weight = A
            elif cell_method[Zcenter] == "mean":
                suffix = "_bounds" if "_bounds" in v else ""
                h = ds[f"thkcello{suffix}"]
                weight = A*h
            else:
                weight = A
            da = (da*weight).coarsen(dim=cdim).sum() / weight.coarsen(dim=cdim).sum()
            da = da.where(da!=0) if v not in coord_vars else da
            da.attrs = attrs

        if v in ds.data_vars:
            ds_coarse[v] = da
        elif v in ds.coords:
            ds_coarse = ds_coarse.assign_coords({v: da})

    ds_coarse = subsample_geocoords(ds_coarse, ds, grid, dim)

    return ds_coarse

def subsample_geocoords(ds_coarse, ds, grid, dim):
    sx = dim["X"]
    sy = dim["Y"]

    ## corner coords
    sx0, sy0 = 0, 0
    ds_coarse = ds_coarse.assign_coords({
        "geolon_c": ds.geolon_c.sel(xq=ds.xq[sx0::sx], yq=ds.yq[sy0::sy]),
        "geolat_c": ds.geolat_c.sel(xq=ds.xq[sx0::sx], yq=ds.yq[sy0::sy]),
    })

    xdim = "xq" if (sx%2==0) else "xh"    
    ydim = "yq" if (sy%2==0) else "yh"

    ## u-velocity coords
    sx0, sy0 = 0, sy//2
    if (sx%2==0) & (sy%2==0):
        lonv,latv = "geolon_c", "geolat_c"
    elif (sx%2==0) & (sy%2==1):
        lonv,latv = "geolon_u", "geolat_u"
    elif (sx%2==1) & (sy%2==0):
        lonv,latv = "geolon_c", "geolat_c"
    elif (sx%2==1) & (sy%2==1):
        lonv,latv = "geolon_u", "geolat_u"
        
    ds_coarse = ds_coarse.assign_coords({
        "geolon_u": xr.DataArray(
            ds[lonv].sel({"xq":ds.xq[sx0::sx], ydim:ds[ydim][sy0::sy]}).values,
            dims = ("yh", "xq",)
        ),
        "geolat_u": xr.DataArray(
            ds[latv].sel({"xq":ds.xq[sx0::sx], ydim:ds[ydim][sy0::sy]}).values,
            dims = ("yh", "xq",)
        )
    })

    ## u-velocity coords
    sx0, sy0 = sx//2, 0
    if (sx%2==0) & (sy%2==0):
        lonv,latv = "geolon_c", "geolat_c"
    elif (sx%2==0) & (sy%2==1):
        lonv,latv = "geolon_c", "geolat_c"
    elif (sx%2==1) & (sy%2==0):
        lonv,latv = "geolon_v", "geolat_v"
    elif (sx%2==1) & (sy%2==1):
        lonv,latv = "geolon_v", "geolat_v"
        
    ds_coarse = ds_coarse.assign_coords({
        "geolon_v": xr.DataArray(
            ds[lonv].sel({xdim:ds[xdim][sx0::sx], "yq":ds.yq[sy0::sy]}).values,
            dims = ("yq", "xh")
        ),
        "geolat_v": xr.DataArray(
            ds[latv].sel({xdim:ds[xdim][sx0::sx], "yq":ds.yq[sy0::sy]}).values,
            dims = ("yq", "xh",)
        )
    })

    ## tracer coords
    sx0, sy0 = sx//2, sy//2
    if (sx%2==0) & (sy%2==0):
        lonv,latv = "geolon_c", "geolat_c"
    elif (sx%2==0) & (sy%2==1):
        lonv,latv = "geolon_u", "geolat_u"
    elif (sx%2==1) & (sy%2==0):
        lonv,latv = "geolon_v", "geolat_v"
    elif (sx%2==1) & (sy%2==1):
        lonv,latv = "geolon", "geolat"

    ds_coarse = ds_coarse.assign_coords({
        "geolon": xr.DataArray(
            ds[lonv].sel({xdim:ds[xdim][sx0::sx], ydim:ds[ydim][sy0::sy]}).values,
            dims = ("yh", "xh",)
        ),
        "geolat": xr.DataArray(
            ds[latv].sel({xdim:ds[xdim][sx0::sx], ydim:ds[ydim][sy0::sy]}).values,
            dims = ("yh", "xh",)
        )
    })

    for c in ds_coarse.coords:
        ds_coarse.coords[c].attrs = ds.coords[c].attrs
    
    return ds_coarse