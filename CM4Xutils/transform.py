"""Conservative vertical remapping (e.g. depth `z` -> potential density `sigma2`).

`remap_vertical_coord` is the core routine: it uses each variable's ``cell_methods``
attribute to decide how to remap (extensive quantities are transformed directly;
intensive ones are thickness-weighted, transformed, then normalized by the remapped
thickness). `itp_tracer_to_transports` is a helper that interpolates the target
density onto cell faces so that transports `umo`/`vmo` can be remapped consistently.
"""

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
        # Conservatively bin `da` onto the target `{coord}_i` interfaces using
        # `target_coord` (the target density evaluated at the source Z interfaces),
        # then relabel the resulting interface axis as the layer-center axis.
        # NaNs are filled with 0 on the way in and out so binning treats dry/empty
        # cells as zero content rather than propagating NaNs.
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

    # Process thkcello (and its snapshot) first: intensive variables are
    # normalized by the *remapped* thickness below, so it must already exist
    # in `ds_trans` by the time those variables are handled.
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

            # Forward-fill the target density through dry (NaN) cells so it stays
            # defined below topography, then interpolate from layer centers to the
            # cell interfaces where the conservative transform expects it.
            target_coord = ds[f"{coord}{suffix}"].ffill(dim=Z_l, limit=None)
            zcoord_at_interface = (
                grid.interp(target_coord, "Z", boundary="extend").chunk({Z_i: -1})
            )
            ds_trans[v] = transform_to_target_coord(da, zcoord_at_interface)

            if Z_cell_method == "mean":
                # Convert back to intensive "mean" quantity
                h = ds_trans[f"thkcello{suffix}"].fillna(0.)
                ds_trans[v] = (ds_trans[v]/h).where(ds_trans[v]!=0.)

    # Transports live on cell faces, not tracer centers, so the target density
    # must first be interpolated onto the u- and v-faces before remapping umo/vmo.
    if all([v in ds.data_vars for v in ["umo", "vmo"]]):
        coord_X, coord_Y = itp_tracer_to_transports(
            grid,
            ds[coord],
            ds.umo,
            ds.vmo
        )
        # Fill dry faces vertically, then interpolate the face density to interfaces.
        coord_X_filled = coord_X.ffill(dim=Z_l, limit=None)
        coord_Y_filled = coord_Y.ffill(dim=Z_l, limit=None)

        ds[f"{coord}_u"] = grid.interp(
            coord_X_filled,
            "Z",
            boundary="extend"
        ).chunk({Z_i: -1})
        # `.where(density)` keeps only truthy (nonzero, non-NaN) densities: real
        # ocean sigma2 is never 0, so this masks out the artificial 0-fill on land.
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

    # Re-assign variable attributes, rewriting the vertical cell method key from
    # the source axis (e.g. "z_l") to the new target axis (e.g. "sigma2_l").
    for v in ds_trans.data_vars:
        ds_trans[v].attrs = ds[v].attrs
        cell_methods_dict = parse_cell_methods(ds_trans[v].cell_methods)
        ds_trans[v].attrs["cell_methods"] = stringify_cell_methods_dict(
            {k.replace(Z_l, f"{coord}_l"):v for (k,v) in cell_methods_dict.items()}
        )

    # Carry over coordinate attributes that the transform dropped.
    for c in ds_trans.coords:
        if ds_trans.coords[c].attrs == {}:
            ds_trans.coords[c].attrs = ds.coords[c].attrs

    return ds_trans

def itp_tracer_to_transports(grid, tracer, transport_X, transport_Y):
    """Interpolate a tracer from cell centers onto the u- and v-transport faces.

    The conservative remapping of `umo`/`vmo` needs the target coordinate (e.g.
    `sigma2`) evaluated on the *faces* where the transports live, not on the tracer
    cell centers. This averages the tracer of the two cells adjacent to each face,
    returning NaN wherever either neighbor is dry so that face densities are only
    defined where the transport itself is (this NaN-handling was the v1.2.0 fix).

    Parameters
    ----------
    grid : `xgcm.Grid`
    tracer : `xr.DataArray` on tracer cell centers (xh, yh)
    transport_X : `xr.DataArray` u-transport on the x-faces (xq, yh)
    transport_Y : `xr.DataArray` v-transport on the y-faces (xh, yq)

    Returns
    -------
    (tracer_X, tracer_Y) : the tracer averaged onto the x-faces and y-faces
    """
    # --- x-faces: average each cell center with its neighbor to the "right" ---
    xc = grid.axes['X'].coords['center']
    xo = grid.axes['X'].coords['outer']
    tracer_right = tracer.rename({xc:xo}).assign_coords({xo:transport_X[xo][1:]})
    tracer_right = xr.where(
        np.logical_and(~np.isnan(tracer_right.roll({xo:-1})), ~np.isnan(tracer_right)),
        0.5*sum([tracer_right.roll({xo:-1}), tracer_right]),
        np.nan
    )
    tracer_X = xr.concat([
        tracer_right.isel({xo:[-1]}).assign_coords(
            {xo:xr.DataArray(transport_X[xo][[0]], dims=(xo,))}
        ),
        tracer_right
    ], dim=xo).assign_coords(transport_X.coords).chunk({xo:-1})

    # --- y-faces: same construction, averaging across the Y axis instead ---
    yc = grid.axes['Y'].coords['center']
    yo = grid.axes['Y'].coords['outer']
    tracer_right = tracer.rename({yc:yo}).assign_coords({yo:transport_Y[yo][1:]})
    tracer_right = xr.where(
        np.logical_and(~np.isnan(tracer_right.roll({yo:-1})), ~np.isnan(tracer_right)),
        0.5*sum([tracer_right.roll({yo:-1}), tracer_right]),
        np.nan
    )
    tracer_Y = xr.concat([
        tracer_right.isel({yo:[-1]}).assign_coords(
            {yo:xr.DataArray(transport_Y[yo][[0]], dims=(yo,))}
        ),
        tracer_right
    ], dim=yo).assign_coords(transport_Y.coords).chunk({yo:-1})

    return tracer_X, tracer_Y