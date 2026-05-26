# %%
import glob
import os
from const import *

"""
Directory structure
{data_dir}/data 
    /original
        /var
        /land
        /axl
    /processed_org_data 
        /annual_per_area_unit
        /land
        /mk_trends_map
    /sensitivity_als
        /CMIP6
            /input_data # resampled data from annual_per_area_unit
            /contribution_mk 
        /VISIT
            /input_data
            /contribution_mk
                /mk_1x1.25
                /mk_0.5x0.5
    /gfdl_esm4_latlon
    /visit_latlon
"""

DATA_SERVER = f"/mnt/dg3/ngoc/cmip6_bvoc_als/data/"
DATA_LOCAL = "../data/"
CMIP6_SENSALS_DIR = f"{DATA_SERVER}/sensitivity_als/CMIP6/"
VISIT_SENSALS_DIR = f"{DATA_SERVER}/sensitivity_als/VISIT/"

PLT_DATA_DIR = f"{CMIP6_SENSALS_DIR}/plt_data"
SUB_PLT_DATA_DIR = [
    "clim_df_rate",
    "clim_rates_ts",
    "ctb_clim_map",
    "ctb_main_map",
    "main_df_rate",
    "main_rates_ts",
    "df_sim",
    "sensitivity_ds",
]


def remove_all_plt_data(folder_path=PLT_DATA_DIR):
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)  # Remove file
            print(file_path)


def make_plt_data_dir():

    if not os.path.exists(PLT_DATA_DIR):
        os.makedirs(PLT_DATA_DIR)

    for sd in SUB_PLT_DATA_DIR:
        new_dir = f"{PLT_DATA_DIR}/{sd}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(new_dir)


DATA_DIR = DATA_LOCAL
if os.path.exists(DATA_SERVER):
    DATA_DIR = DATA_SERVER

VAR_DIR = os.path.join(DATA_DIR, "original/var")
LAND_DIR = os.path.join(DATA_DIR, "original/land")
AXL_DIR = os.path.join(DATA_DIR, "original/axl")

LIST_ATTR = [attr.split("\\")[-1] for attr in glob.glob(os.path.join(VAR_DIR, "*"))]

ISOP_LIST = glob.glob(os.path.join(VAR_DIR, "emiisop", "*.nc"))
BVOC_LIST = glob.glob(os.path.join(VAR_DIR, "emibvoc", "*.nc"))

AREA_LIST = glob.glob(os.path.join(AXL_DIR, VAR_AREA, "*.nc"))
SFLTF_LIST = glob.glob(os.path.join(AXL_DIR, VAR_SFTLF, "*.nc"))
MASK_LIST = glob.glob(os.path.join(AXL_DIR, VAR_MASK, "*.nc"))


VISIT_LAT_FILE = os.path.join(DATA_DIR, "visit_latlon", "visit_lat.npy")
VISIT_LONG_FILE = os.path.join(DATA_DIR, "visit_latlon", "visit_long.npy")
GFDL_LAT_FILE = os.path.join(DATA_DIR, "gfdl_esm4_latlon", "lat.npy")
GFDL_LONG_FILE = os.path.join(DATA_DIR, "gfdl_esm4_latlon", "lon.npy")

# %%
