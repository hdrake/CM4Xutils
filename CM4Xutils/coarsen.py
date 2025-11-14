from .grid_preprocess import *
from .version import __version__

def horizontally_coarsen(ds, grid, dim, skip_coords=False):
    """Horizontally coarsen a dataset by given coarsening factors.

    Grid-aware coarsening, which uses the `cell_methods` attributes
    of variables to determine whether a variable should be
    summed, averaged, or subsampled in a particular dimension.

    Parameters
    ----------
    ds : `xr.Dataset` containing coordinates and variables.
    grid : `xgcm.Grid` object with axis and metric metadata
    dim : mapping of horizontal axes to coarsening factors
        A dictionary mapping the horizontal axes ("X" and "Y")
        to the corresponding integer coarsening factor.
        Currently requires coarsening factors to be divisors of
        the corresponding dimension.
        Example: `dim = {"X":2, "Y":2}`
    skip_coords (bool) : Whether to skip subsampling of coordinates

    Returns
    -------
    ds_coarse : horizontally coarsened `xr.Dataset`
    """
    ds_coarse = xr.Dataset(attrs=ds.attrs)

    coarsen_key_coords = ["areacello", "deptho", "wet", "wet_u", "wet_v", "dxCv", "dyCu"]
    coord_vars = [
        e for e in coarsen_key_coords
        if (e in ds.coords)
    ]

    var_list = list(ds.data_vars) + coord_vars
    for v in var_list:

        da = ds[v].copy()

        if "cell_methods" not in da.attrs:
            print(f"Skipping variable {v} because `cell_methods` attribute not defined.")
            continue
            
        cell_method = parse_cell_methods(da.cell_methods)
        try:
            dim_names = {
                "X": grid._get_dims_from_axis(da, "X")[0],
                "Y": grid._get_dims_from_axis(da, "Y")[0]
            }
        except:
            print(f"Skipping {v} because independent of 'X' and 'Y' dims.")
            continue

        partially_wet_mask = (ds.wet.fillna(0.) > 0.).persist()
        wet_pos = ds.wet.where(ds.wet > 0.).persist()

        # First catch variables that are averaged in both X and Y
        if all([cell_method[dim_var] == "mean" for dim_var in dim_names.values()]):
            # Weighted-average using either:
            # 1. ocean cell surface area, for variables with cell methods that are
            #    {X:mean, Y:mean} in 2D or additionally with {Z:sum} in 3D.
            #    E.g. `tos` (surface ocean temperature), thkcello, and tracer budget tendencies
            # 2. ocean cell volume (surface area times instantaneous or time-mean thickness),
            #    for variables with 3D cell methods {X:mean, Y:mean, Z:mean}.
            #    E.g. tracers like `thetao`, `cfc11`, etc.
            # 3. total cell surface area (ocean + land), as a special case for the `wet` mask,
            #    so that we can divide the coarsened ocean cell area (`areacello` by `wet`) to
            #    recover the total cell area.

            # Case 1. Default to ocean surface area weights
            A = grid.get_metric(da, ["X", "Y"]).fillna(0.) # ocean cell area
            weight = A.where(partially_wet_mask, 0.).persist() # ensure that land area is masked
            # Case 3. Overwrite weight with total cell area when calculating wet mask
            if v == "wet":
                # convert ocean area to total cell area
                weight = xr.where(
                    partially_wet_mask,
                    A/wet_pos,
                    A
                ).fillna(0.).persist()
            elif "Z" in grid.axes:
                Zcenter = grid.axes["Z"].coords["center"]
                if Zcenter in cell_method:
                    if cell_method[Zcenter] == "mean":
                        # Case 2. Overwrite weights with ocean volume,
                        # calculated by multiplying ocean cell area by
                        # mean ocean cell thickness
                        suffix = "_bounds" if "_bounds" in v else ""
                        if f"volcello{suffix}" in ds:
                            weight = ds[f"volcello{suffix}"]
                        else:
                            h = ds[f"thkcello{suffix}"]
                            weight = A*h
                    
            cdim = {dim_var:dim[dim_name] for (dim_name, dim_var) in dim_names.items()}
            attrs = da.attrs
            da_weighted_integral = (da*weight).fillna(0.).coarsen(dim=cdim).sum()
            weight_integral = weight.fillna(0.).coarsen(dim=cdim).sum()
            da_coords = {
                c:da_weighted_integral[c] for c in da_weighted_integral.coords
                if c in weight_integral.coords
            }
            weight_integral = weight_integral.assign_coords(da_coords)
            da = da_weighted_integral / weight_integral.where(weight_integral != 0.)
            if v == "wet":
                da = da.round(5)
            da = da if (v in coord_vars) else da.where(da!=0.)
            da.attrs = attrs
            
        elif all([cell_method[dim_var] == "sum" for dim_var in dim_names.values()]):
            # Only case should be grid cell area (`areacello`)
            weight = xr.where(
                partially_wet_mask,
                partially_wet_mask.astype("float"),
                0.
            ).persist()
            cdim = {dim_var:dim[dim_name] for (dim_name, dim_var) in dim_names.items()}
            attrs = da.attrs
            da = (da*weight).fillna(0.).coarsen(dim=cdim).sum()
            da = xr.where(
                da == 0.,
                ds[v].fillna(0.).coarsen(dim=cdim).sum(), # land cells should have land area
                da
            )
            da = da if (v in coord_vars) else da.where(da!=0.)
            da.attrs = attrs
            
        else:
            # First check if we need to take an average in one of the directions,
            # because we need to use the cell widths at full resolution
            if any([cell_method[dim_var] == "mean" for dim_var in dim_names.values()]):
                for (dim_name, dim_var) in dim_names.items():
                    cdim = {dim_var:dim[dim_name]}
                    if cell_method[dim_var] == "mean":
                        if v == "wet_u":
                            attrs = da.attrs
                            partially_wet_u_mask = ds.wet_u.fillna(0.) > 0.
                            weight = xr.where(
                                partially_wet_u_mask,
                                ds.dyCu.fillna(0.)/ds.wet_u.where(ds.wet_u > 0),
                                ds.dyCu
                            ).fillna(0.)
                            da_weighted_integral = (da*weight).fillna(0.).coarsen(dim=cdim).sum()
                            weight_integral = weight.fillna(0.).coarsen(dim=cdim).sum()
                            da_coords = {
                                c:da_weighted_integral[c] for c in da_weighted_integral.coords
                                if c in weight_integral.coords
                            }
                            weight_integral = weight_integral.assign_coords(da_coords)
                            da = (da_weighted_integral / weight_integral.where(weight_integral != 0.)).round(5)
                            da.attrs = attrs
                        elif v == "wet_v":
                            attrs = da.attrs
                            partially_wet_v_mask = ds.wet_v.fillna(0.) > 0.
                            weight = xr.where(
                                partially_wet_v_mask,
                                ds.dxCv.fillna(0.)/ds.wet_v.where(ds.wet_v > 0),
                                ds.dxCv
                            ).fillna(0.)
                            da_weighted_integral = (da*weight).fillna(0.).coarsen(dim=cdim).sum()
                            weight_integral = weight.fillna(0.).coarsen(dim=cdim).sum()
                            da_coords = {
                                c:da_weighted_integral[c] for c in da_weighted_integral.coords
                                if c in weight_integral.coords
                            }
                            weight_integral = weight_integral.assign_coords(da_coords)
                            da = (da_weighted_integral / weight_integral.where(weight_integral != 0.)).round(5)
                            da.attrs = attrs
                        else:
                            print(f"Skipping {v} because cell method is mean in {dim_var}.")
                            continue

            # Second check if we need to take a sum in either direction
            elif any([cell_method[dim_var] == "sum" for dim_var in dim_names.values()]):
                for (dim_name, dim_var) in dim_names.items():
                    cdim = {dim_var:dim[dim_name]}
                    if cell_method[dim_var] == "sum":
                        if dim_name == "X":
                            partially_wet_mask = ds.wet_v.fillna(0.) > 0.
                            weight = xr.where(
                                partially_wet_mask,
                                partially_wet_mask.astype("float"),
                                0.
                            )
                        elif dim_name == "Y":
                            partially_wet_mask = ds.wet_u.fillna(0.) > 0.
                            weight = xr.where(
                                partially_wet_mask,
                                partially_wet_mask.astype("float"),
                                0.
                            )
                        attrs = da.attrs
                        da = (da*weight).fillna(0.).coarsen(dim=cdim).sum()
                        da = xr.where(
                            da == 0.,
                            ds[v].fillna(0.).coarsen(dim=cdim).sum(), # land-only faces should have land width
                            da
                        )
                        da = da if (v in coord_vars) else da.where(da!=0.)
                        da.attrs = attrs

            # Finally, check if we need to do just simple subsampling
            for (dim_name, dim_var) in dim_names.items():
                if cell_method[dim_var] == "point":
                    da = da.sel({dim_var: da[dim_var][::dim[dim_name]]})

        if v in ds.data_vars:
            ds_coarse[v] = da
            
        elif v in ds.coords:
            ds_coarse = ds_coarse.assign_coords({v: xr.DataArray(da.values, dims=da.dims)})

    if not(skip_coords):
        ds_coarse = subsample_geocoords(ds_coarse, ds, grid, dim)

    for c in coord_vars:
        ds_coarse[c].attrs = ds[c].attrs
        if c == "areacello":
            ds_coarse[c].attrs["note"] = (
                "We ignore land cells in partially wet cells when coarsening, "
                "so that tracer content can be accurately reconstructed "
                "by multiplying coarsened area-averaged tendencies by it. "
                "Fully wet (`wet==1.0`) and fully dry (`wet==0.0`) cells "
                "should be unaffected, and will just represent the total "
                "cell area. For the partially wet cells, total cell area can "
                "be derived from the ocean area by divding `areacello` by `wet`."
            )
        elif c == "wet":
            ds_coarse[c].attrs["note"] = (
                "Can be between 0 and 1 if coarse cell includes both wet "
                "and dry sub-cells."
            )
        elif c in ["wet_u", "wet_v"]:
            ds_coarse[c].attrs["note"] = (
                "Can be between 0 and 1 if coarse face includes both wet "
                "and dry sub-faces."
            )

    coarsening_comment = f"""Diagnostics have been conservatively coarsened by Henri F. Drake
(hfdrake@uci.edu) using the CM4Xutils python package v{__version__}
(https://github.com/hdrake/CM4Xutils) and with coarsening factors of {dim}. """
    if "provenance" in ds_coarse.attrs:
        ds_coarse.attrs["provenance"] += coarsening_comment
    else: 
        ds_coarse.attrs["provenance"] = coarsening_comment
        
    return ds_coarse

def subsample_geocoords(ds_coarse, ds, grid, dim):
    """Subsample horizontal coordinates according to coarsening factors.

    Overwrites a coarsened dataset's coordinates with the correct
    subsampling of coordinates from the original dataset.

    Parameters
    ----------
    ds_coarse : coarsened `xr.Dataset`, derived from `ds`
    ds : original `xr.Dataset`
    grid : original `grid`
    dim : mapping of horizontal axes to coarsening factors
        A dictionary mapping the horizontal axes ("X" and "Y")
        to the corresponding integer coarsening factor.
        Currently requires coarsening factors to be divisors of
        the corresponding dimension.
        Example: `dim = {"X":2, "Y":2}`
    """
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

    ds_coarse = ds_coarse.assign_coords({c: xr.DataArray(np.arange(0, ds_coarse[c].size), dims=(c,)) for c in ["xh", "yh", "xq", "yq"]})
    ds_coarse["xh"].attrs = {"long_name":"cell center x-index (nominally longitude)", "cell_methods":"xh:point"}
    ds_coarse["yh"].attrs = {"long_name":"cell center y-index (nominally latitude)",  "cell_methods":"yh:point"}
    ds_coarse["xq"].attrs = {"long_name":"cell corner x-index (nominally longitude)", "cell_methods":"xq:point"}
    ds_coarse["yq"].attrs = {"long_name":"cell corner y-index (nominally latitude)",  "cell_methods":"yq:point"}
    
    return ds_coarse