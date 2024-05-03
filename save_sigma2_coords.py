import CM4Xutils

ds = CM4Xutils.load_averages_and_snapshots(
    "odiv-230", coord="rho2", time="010101*"
)

ds[['sigma2_l', 'sigma2_i']].to_netcdf("data/sigma2_coords.nc", mode="w")