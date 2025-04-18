import doralite
import gfdl_utils.core as gu
import CM4Xutils

import subprocess
import os
import sys
import glob
import time as time_module

collab_path = "/collab1/data_untrusted/Henri.Drake/CM4X/"

key_vars = {
    # 3D (vertical, lat, lon) monthly
    ## STORAGE ESTIMATE: 20 GB/5yr/var x 70 periods x 2.5 exp x 3 var = 12.5 GB
    "ocean_month_rho2": ["umo", "vmo", "thkcello"],
    ## STORAGE ESTIMATE: 15 GB/5yr/var x 70 periods x 4 exp x 2 var = 8400 GB
    "ocean_month_z": ["so", "thetao"],
    ## STORAGE ESTIMATE: 200 MB/5yr/var x 70 periods x 4 exp x 7 var = 168 GB
    "atmos_cmip": ["ua", "va", "ta", "hus", "zg", "clt", "psl"],
    # 2D variables (lat, lon)
    ## STORAGE ESTIMATE: 0.5 GB/5yr/var x 70 periods x 4 exp x 15 = 2100 GB
    "ocean_inert_month": [
        'sf6_alpha', 'sf6_csurf', 'sf6_sc_no', 'sf6_stf_gas', 'cfc11_alpha',
        'cfc11_csurf', 'cfc11_sc_no', 'cfc12_alpha', 'cfc12_csurf', 'cfc12_sc_no',
        'cfc12_stf_gas', 'fgcfc11', 'fgcfc12', 'fgsf6', 'cfc11_stf_gas'
    ],
    ## STORAGE ESTIMATE: 1000 GB
    "ocean_monthly": [
        "wfo", "fsitherm", "prlq", "hfds", "zos",
        "taux", "tauy", "pbo", "sob", "tob", "ePBL_h_ML"
    ],
    "atmos": {
    ## STORAGE ESTIMATE: 1.2 GB/5yr/var x 70 periods x 4 exp x 3 var = 1000 GB
        "daily/5yr": ["tas", "ps", "precip"],
    ## STORAGE ESTIMATE: 100 GB
        "monthly/5yr": ["netrad_toa", "olr", "ps", "omega", "tas", "precip", "alb_sfc"]
    },
    ## STORAGE ESTIMATE: 200 GB
    "ice": ["siconc", "siu", "siv", "sivol", "RAIN"],
    ## STORAGE ESTIMATE: 200 GB
    "bergs_month": ['virtual_area', 'melt', 'real_calving', 'accum_calving', 'mass'],
    ## ONLY SAVED FOR LAST 10 YEARS OF EXPERIMENTS!
    ## STORAGE ESTIMATE: 20 GB/5yr/var x 2 periods x 2 exp x 6 variables = 480 GB
    "ocean_daily": ["sos", "tos", "ssu", "ssv", "zos", 'omldamax'], 
    # 3D (vertical, lat, lon) annual (1x1 degree)
    ## STORAGE ESTIMATE: 20 GB
    "ocean_annual_z_d2_1x1deg": [
        "Kd_BBL", "Kd_ePBL", "Kd_interface", "Kd_itides", "Kd_shear", "obvfsq"
    ],
    # 2D variables (lat, lon) (1x1 degree)
    ## STORAGE ESTIMATE: 20 GB
    "ocean_monthly_1x1deg": None, 
    ## STORAGE ESTIMATE: 20 GB
    "ice_1x1deg": None,
    # 0D variables
    "ocean_scalar_annual": None,
    "ocean_scalar_monthly": None,
    "atmos_scalar": None,
}

def call_gcp(source, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    try:
        # Run the gcp command
        result = subprocess.run(['gcp', source, destination], 
                                check=True, 
                                capture_output=True, 
                                text=True,
                                timeout=1800)
        print("File copied successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while copying: {e}")
        print(f"gcp stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def copy_CM4Xp125_to_collab(ppname_list = None, dmget=True, gcp=True):
    """Copy CM4Xp125 diagnostics to collab Globus endpoint."""
    for k,odiv in CM4Xutils.exp_dict["CM4Xp125"].items():
        if k=="hgrid":
            continue
        if "continued" in k:
            continue
        
        pp = doralite.dora_metadata(odiv)["pathPP"]
        all_vars = gu.get_allvars(pp)
        
        if ppname_list is None:
            var_dict = key_vars.copy()
            
        elif isinstance(ppname_list, list):
            var_dict = {k:key_vars[k] for k in ppname_list}

        for ppname, v_list in var_dict.items():
            if ("inert" in ppname) & ("spinup" in k):
                continue

            out = "ts"
            local = gu.get_local(pp, ppname, out)
            v_list = v_list if v_list is not None else all_vars[ppname]
            v_dict = v_list if ppname == "atmos" else {local: v_list}
            if ppname == "ocean_daily":
                t = "034*" if "piControl" in k else "209*"
                if ("spinup" in k) or ("continued" in k):
                    continue
            elif "spinup" in k:
                t = "00*"
            else:
                t = "*"

            all_paths = ""
            for local, v_list in v_dict.items():
                print("Files to be copied:")
                paths = []
                for v in v_list:
                    path = gu.get_pathspp(pp,ppname,out,local,t,v)
                    print(path, end="\n")
                    paths += glob.glob(path)
                    for s in glob.glob(path):
                        all_paths += s+" "
                paths = sorted(paths)
        
                if dmget:
                    print("Issuing dmget command to migrate data to disk:")
                    gu.issue_dmget(paths)
                    all_on_disk = False
                    while not(all_on_disk):
                        all_on_disk = gu.query_all_ondisk(paths)
                        time_module.sleep(0.01)
                    print("Migration complete.", end="\n\n")

                if gcp:
                    for source_file in paths:
                        destination_path = collab_path + "/".join(source_file.split("/")[6:-1])
                        destination_file = f"{destination_path}/{source_file.split("/")[-1]}"
                        print(f"Copying:\n From: '{source_file}'\n To: '{destination_file}'")
                        success = call_gcp(source_file, destination_file)
                        print("\n")
                        time_module.sleep(0.01)

            if not(dmget) and not(gcp):
                bash_script_content = f"""#!/bin/bash
echo "Issuing dmget commands for CM4Xp125-{k} {ppname}!"
dmget {all_paths}
"""

                bash_script_name = f"dmget_CM4Xp125-{k}_{ppname}.sh"
                with open(bash_script_name, "w") as bash_file:
                    bash_file.write(bash_script_content)

                print(f"Bash script {bash_script_name} has been created successfully!")
