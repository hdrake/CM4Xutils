from .grid_preprocess import *

def remap_to_sigma2(ds, grid):

    def transform_to_sigma2(da, sigma2_target_data):
        return (
            grid.transform(
                da.fillna(0.),
                "Z",
                target = ds.sigma2_i,
                target_data = sigma2_target_data,
                method="conservative",
            ).fillna(0.)
            .rename({"sigma2_i": "sigma2_l"})
            .assign_coords({"sigma2_l": ds.sigma2_l})
        )

    Z_l = grid.axes["Z"].coords["center"]
    Z_i = grid.axes["Z"].coords["outer"]
    
    ds_sigma2 = xr.Dataset(coords=ds.drop_dims([Z_l, Z_i]).coords, attrs=ds.attrs)
    ds_sigma2.attrs["provenance"] = """Diagnostics have been conservatively remapped into monthly-mean
sigma2 coordinates by Henri F. Drake (hfdrake@uci.edu) using the
CM4Xutils python package (https://github.com/hdrake/CM4Xutils)."""

    data_vars = (
        ["thkcello", "thkcello_bounds"] +
        [v for v in ds.data_vars if "thkcello" not in v]
    )
    
    for v in data_vars:
        if (
            (v not in ["sigma2", "umo", "vmo"]) & # these already covered
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

            sigma2_at_interface = (
                grid.interp(ds[f"sigma2{suffix}"], "Z", boundary="extend").chunk({Z_i: -1})
            )
            ds_sigma2[v] = transform_to_sigma2(da, sigma2_at_interface)

            if Z_cell_method == "mean":
                # Convert back to intensive "mean" quantity
                h = ds_sigma2[f"thkcello{suffix}"].fillna(0.)
                ds_sigma2[v] = (ds_sigma2[v]/h).where(ds_sigma2[v]!=0.)

    ds["sigma2_u"] = grid.interp(
        grid.interp(ds.sigma2, "X"),
        "Z",
        boundary="extend"
    ).chunk({Z_i: -1})
    ds_sigma2["umo"] = transform_to_sigma2(ds.umo, ds[f"sigma2_u"])

    ds["sigma2_v"] = grid.interp(
        grid.interp(ds.sigma2, "Y"),
        "Z",
        boundary="extend"
    ).chunk({Z_i: -1})
    ds_sigma2["vmo"] = transform_to_sigma2(ds.vmo, ds[f"sigma2_v"])


    # Re-assign attributes
    for v in ds_sigma2.data_vars:
        ds_sigma2[v].attrs = ds[v].attrs
        cell_methods_dict = parse_cell_methods(ds_sigma2[v].cell_methods)
        ds_sigma2[v].attrs["cell_methods"] = stringify_cell_methods_dict(
            {k.replace(Z_l, "sigma2_l"):v for (k,v) in cell_methods_dict.items()}
        )

    for c in ds_sigma2.coords:
        if ds_sigma2.coords[c].attrs == {}:
            ds_sigma2.coords[c].attrs = ds.coords[c].attrs
    
    return ds_sigma2

