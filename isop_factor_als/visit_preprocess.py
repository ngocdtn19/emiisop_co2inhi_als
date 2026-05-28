# %%
import xarray as xr
import cftime
import numpy as np
import copy

from datetime import datetime, timedelta

# from CMIP6Var import CMIP6Var

from .const import *
from .mypath import *


def year_2_cft(dcm_year):
    """Convert the decimal year to cftime format
    Param:
        decimal year
    Return:
        cftime
    """
    year = int(dcm_year)
    rem = dcm_year - year

    base = datetime(year, 1, 1)
    dt = base + timedelta(
        seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
    )

    cft = cftime.datetime(dt.year, dt.month, dt.day)

    return cft


def visit_t2cft(visit_nc, var_name):
    """Convert the VISIT data's decimal year to cftime format for consistency with CMIP6 data
        Rename to emiisop and extract data over 1850-2014 as same as CMIP6 data
    Param:
        visit_nc: VISIT netcdf files
        var_name: variable name
    Return:
        xarray.Dataset: original VISIT ds with format same as original CMIP6 data
    """
    org_visit_ds = xr.open_dataset(visit_nc, decode_times=False)
    cft = [year_2_cft(org_time) for org_time in org_visit_ds.time.values]
    org_visit_ds.coords["time"] = cft

    org_visit_ds = org_visit_ds.rename({list(org_visit_ds.data_vars)[0]: var_name})

    org_visit_ds = org_visit_ds.where(
        org_visit_ds[var_name].sel(time=slice("2000-01", "2023-12"))
    )
    org_visit_ds = org_visit_ds.where(org_visit_ds[var_name] != -9999.0)
    org_visit_ds = org_visit_ds.where(org_visit_ds[var_name] != -99999.0)
    return org_visit_ds.groupby("time.year").sum("time")


def grid_area(lat1, lat2, lon1, lon2):
    """Calculate a grid area from lat, lon
    Param:
        lat1, lat2, lon1, lon2: latitude and longtitude of grid
    Return:
        float: area
    """
    E_RAD = 6378137.0
    # m, GRS-80(revised)
    E_FLAT = 298.257
    PI = 3.1415926
    E_EXC = math.sqrt(2.0 / E_FLAT - 1.0 / (E_FLAT * E_FLAT))

    if lat1 > 90.0:
        lat1 = 90.0
    if lat2 < -90.0:
        lat2 = -90.0

    m_lat = (lat1 + lat2) / 2.0 * PI / 180.0

    aa1 = 1.0 - E_EXC * E_EXC * math.sin(m_lat) * math.sin(m_lat)
    l_lat = (
        PI
        / 180.0
        * E_RAD
        * (1.0 - E_EXC * E_EXC)
        / math.pow(aa1, 1.5)
        * math.fabs(lat1 - lat2)
    )

    aa2 = 1.0 - E_EXC * E_EXC * math.sin(lat1 * PI / 180.0) * math.sin(
        lat1 * PI / 180.0
    )
    l_lon1 = (
        PI
        / 180.0
        * E_RAD
        * math.cos(lat1 * PI / 180.0)
        / math.sqrt(aa2)
        * math.fabs(lon1 - lon2)
    )
    aa3 = 1.0 - E_EXC * E_EXC * math.sin(lat2 * PI / 180.0) * math.sin(
        lat2 * PI / 180.0
    )
    l_lon2 = (
        PI
        / 180.0
        * E_RAD
        * math.cos(lat2 * PI / 180.0)
        / math.sqrt(aa3)
        * math.fabs(lon1 - lon2)
    )

    area = (l_lon1 + l_lon2) * l_lat / 2.0

    return area


def cal_ds_area(
    visit_nc=f"{DATA_DIR}/original/var/emiisop/emiisop_AERmon_VISIT-S3(G1997)_historical_r1i1p1f1_gn_170001-202112.nc",
):
    """Calculate grid area of VISIT and write to a netcdf file with same format as CMIP6 data
    Param:
        visit_nc: emiisop data of VISIT in netcdf
    Return:
        xarray.Dataset: grid area
    """
    ds = xr.open_dataset(visit_nc, decode_times=False)
    nlat = ds.lat.values.reshape(-1)
    nlon = ds.lon.values.reshape(-1)
    garea = []
    ds_area = {}

    final_arr = []
    for g in range(0, len(nlat)):
        glat = 89.75 - 0.5 * g
        garea.append(grid_area(glat + 0.25, glat - 0.25, 0.0, 0.5))
    arr = np.array(garea)

    _ = [final_arr.append([i] * len(nlon)) for i in arr]
    data = np.array(final_arr)
    ds_area = xr.Dataset(
        data_vars=dict(areacella=(["lat", "lon"], data)),
        coords=dict(
            lat=nlat,
            lon=nlon,
        ),
    )

    return ds_area


# def save_2_nc(org_visit_ds):
#     var_name = "emiisop"
#     m_name = "VISIT"
#     org_visit_ds.to_netcdf(f"{DATA_DIR}/original/var/{var_name}/{var_name}_AERmon_{m_name}_historical_r1i1p1f1_gn_190101-201512.nc")


# %%
