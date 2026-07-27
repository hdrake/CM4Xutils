#!/usr/bin/env python
# coding: utf-8
"""Regenerate `data/sigma2_coords.nc`, the CM4X 74-layer target density grid.

The target density coordinate is the `rho2` diagnostic coordinate of the
`ocean_month_rho2` pp output, which is identical across CM4X models and
experiments. `sigma2` is that coordinate minus 1000 kg/m3.

`CM4Xutils.add_sigma2_coords` reads the resulting file and pads it with one
extra layer at each end, so this script writes the raw 74-layer grid only.
"""
import xarray as xr

import doralite
import gfdl_utils.core as gu

from CM4Xutils import exp_dict

odiv = exp_dict["CM4Xp25"]["piControl"]
time = "010101*"

pp = doralite.dora_metadata(odiv)["pathPP"]
ppname = "ocean_month_rho2"
out = "ts"
local = gu.get_local(pp, ppname, out)
ds = gu.open_frompp(pp, ppname, out, local, time, "thkcello", dmget=True)

sigma2_coords = xr.Dataset(
    coords={
        "sigma2_i": xr.DataArray(ds.rho2_i.values - 1000., dims=("sigma2_i",)),
        "sigma2_l": xr.DataArray(ds.rho2_l.values - 1000., dims=("sigma2_l",)),
    },
    attrs=ds.attrs
)
sigma2_coords = sigma2_coords.assign_coords({
    "rho2_i": xr.DataArray(ds.rho2_i.values, dims=("sigma2_i",), attrs=ds.rho2_i.attrs),
    "rho2_l": xr.DataArray(ds.rho2_l.values, dims=("sigma2_l",), attrs=ds.rho2_l.attrs),
})

sigma2_coords.to_netcdf("../data/sigma2_coords.nc", mode="w")
