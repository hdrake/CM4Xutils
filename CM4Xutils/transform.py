from .grid_preprocess import *
from .version import __version__

def remap_vertical_coord(coord, ds, grid):
    """"Remap vertical coordinate to target coordinate

    Uses `cell_method` attribute in `ds.data_vars` variables
    to determine how to vertically remap. For extensive 
    variables (e.g. with `"zl:sum" in `cell_method`), we simply
    do a conservative remapping. For intensive variables (e.g.
    with `"zl:mean"`), however, we need to conservatively remap
    both the variable and corresponding cell thicknesses, then
    divide the remapped extensive content by the remapped thickness
    to get the intensive values.

    For `umo` and `vmo`, we interpolate the target tracer concentration
    onto the corresponding cell faces and then do the remapping.

    Parameters
    ----------
    coord : str
        Name of vertical coordinate variable.
        Only tested case is "sigma2".
    ds : `xr.Dataset`
    grid : `xgcm.Grid`

    Returns
    -------
    ds_trans : transformed `xr.Dataset`
    """

    def transform_to_target_coord(da, target_coord):
        return (
            grid.transform(
                da.fillna(0.),
                "Z",
                target = ds[f"{coord}_i"],
                target_data = target_coord,
                method="conservative",
            ).fillna(0.)
            .rename({f"{coord}_i": f"{coord}_l"})
            .assign_coords({f"{coord}_l": ds[f"{coord}_l"]})
        )

    Z_l = grid.axes["Z"].coords["center"]
    Z_i = grid.axes["Z"].coords["outer"]
    
    ds_trans = xr.Dataset(coords=ds.drop_dims([Z_l, Z_i]).coords, attrs=ds.attrs)
    ds_trans.attrs["provenance"] = f"""Diagnostics have been conservatively remapped into monthly-mean
{coord} coordinates by Henri F. Drake (hfdrake@uci.edu) using the
CM4Xutils python package v{__version__} (https://github.com/hdrake/CM4Xutils). """

    data_vars = (
        [v for v in ["thkcello", "thkcello_bounds"] if v in ds.data_vars] +
        [v for v in ds.data_vars if "thkcello" not in v]
    )
    for v in data_vars:
        if (
            (v not in [f"{coord}", "umo", "vmo"]) & # these already covered
            all([d in ds[v].dims for d in ["xh", "yh", Z_l]]) # on tracer grid
        ):
            suffix = "_bounds" if "_bounds" in v else ""
            Z_cell_method = parse_cell_methods(ds[v].cell_methods)[Z_l]
            if Z_cell_method == "mean":
                # Convert to extensive "sum" quantity for conservative binning
                h = ds[f"thkcello{suffix}"].fillna(0.)
                da = ds[v]*h
            else:
                da = ds[v]

            target_coord = fillna_below(grid, ds[f"{coord}{suffix}"])
            zcoord_at_interface = (
                grid.interp(target_coord, "Z", boundary="extend").chunk({Z_i: -1})
            )
            ds_trans[v] = transform_to_target_coord(da, zcoord_at_interface)

            if Z_cell_method == "mean":
                # Convert back to intensive "mean" quantity
                h = ds_trans[f"thkcello{suffix}"].fillna(0.)
                ds_trans[v] = (ds_trans[v]/h).where(ds_trans[v]!=0.)

    if all([v in ds.data_vars for v in ["umo", "vmo"]]):
        coord_X, coord_Y = itp_tracer_to_transports(
            grid,
            ds[coord],
            ds.umo,
            ds.vmo
        )
        coord_X_filled = fillna_below(grid, coord_X)
        coord_Y_filled = fillna_below(grid, coord_Y)
        
        ds[f"{coord}_u"] = grid.interp(
            coord_X_filled,
            "Z",
            boundary="extend"
        ).chunk({Z_i: -1})
        ds_trans["umo"] = transform_to_target_coord(
            ds.umo,
            ds[f"{coord}_u"].where(ds[f"{coord}_u"])
        )

        ds[f"{coord}_v"] = grid.interp(
            coord_Y_filled,
            "Z",
            boundary="extend"
        ).chunk({Z_i: -1})
        ds_trans["vmo"] = transform_to_target_coord(
            ds.vmo,
            ds[f"{coord}_v"].where(ds[f"{coord}_v"])
        )

    # Re-assign attributes
    for v in ds_trans.data_vars:
        ds_trans[v].attrs = ds[v].attrs
        cell_methods_dict = parse_cell_methods(ds_trans[v].cell_methods)
        ds_trans[v].attrs["cell_methods"] = stringify_cell_methods_dict(
            {k.replace(Z_l, f"{coord}_l"):v for (k,v) in cell_methods_dict.items()}
        )

    for c in ds_trans.coords:
        if ds_trans.coords[c].attrs == {}:
            ds_trans.coords[c].attrs = ds.coords[c].attrs
    
    return ds_trans

def itp_tracer_to_transports(grid, tracer, transport_X, transport_Y):
    xc = grid.axes['X'].coords['center']
    xo = grid.axes['X'].coords['outer']
    tracer_right = tracer.rename({xc:xo}).assign_coords({xo:transport_X[xo][1:]})
    tracer_right = xr.where(
        np.logical_and(~np.isnan(tracer_right.roll({xo:-1})), ~np.isnan(tracer_right)),
        0.5*sum([tracer_right.roll({xo:-1}), tracer_right]),
        0.
    )
    tracer_X = xr.concat([
        tracer_right.isel({xo:[-1]}).assign_coords(
            {xo:xr.DataArray(transport_X[xo][[0]], dims=(xo,))}
        ),
        tracer_right
    ], dim=xo).assign_coords(transport_X.coords)

    yc = grid.axes['Y'].coords['center']
    yo = grid.axes['Y'].coords['outer']
    tracer_right = tracer.rename({yc:yo}).assign_coords({yo:transport_Y[yo][1:]})
    tracer_right = xr.where(
        np.logical_and(~np.isnan(tracer_right.roll({yo:-1})), ~np.isnan(tracer_right)),
        0.5*sum([tracer_right.roll({yo:-1}), tracer_right]),
        0.
    )
    tracer_Y = xr.concat([
        tracer_right.isel({yo:[-1]}).assign_coords(
            {yo:xr.DataArray(transport_Y[yo][[0]], dims=(yo,))}
        ),
        tracer_right
    ], dim=yo).assign_coords(transport_Y.coords)

    return tracer_X, tracer_Y

def fillna_below(grid, da):
    
    # First last non-NaN vertical index
    da = da.where(da!=0)
    zc = grid.axes['Z'].coords['center']
    idx = np.isnan(da).argmax(zc)
    idx = xr.where(idx>0, idx-1, idx).compute()

    # Use bottom-most valid point to overwrite NaN points below
    return xr.where(
        da[zc] > da[zc].isel({zc:idx}),
        da.isel({zc:idx}),
        da
    )
