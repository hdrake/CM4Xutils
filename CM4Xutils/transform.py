from .grid_preprocess import *
from .version import __version__

def remap_vertical_coord(coord, ds, grid):

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

            zcoord_at_interface = (
                grid.interp(ds[f"{coord}{suffix}"], "Z", boundary="extend").chunk({Z_i: -1})
            )
            ds_trans[v] = transform_to_target_coord(da, zcoord_at_interface)

            if Z_cell_method == "mean":
                # Convert back to intensive "mean" quantity
                h = ds_trans[f"thkcello{suffix}"].fillna(0.)
                ds_trans[v] = (ds_trans[v]/h).where(ds_trans[v]!=0.)

    if "umo" in ds.data_vars:
        ds[f"{coord}_u"] = grid.interp(
            grid.interp(ds[coord], "X"),
            "Z",
            boundary="extend"
        ).chunk({Z_i: -1})
        ds_trans["umo"] = transform_to_target_coord(ds.umo, ds[f"{coord}_u"])

    if "vmo" in ds.data_vars:
        ds[f"{coord}_v"] = grid.interp(
            grid.interp(ds[coord], "Y"),
            "Z",
            boundary="extend"
        ).chunk({Z_i: -1})
        ds_trans["vmo"] = transform_to_target_coord(ds.vmo, ds[f"{coord}_v"])


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

