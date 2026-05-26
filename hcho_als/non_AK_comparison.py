# %%
import sys

sys.path.append("/home/ngoc/nc2gtool/pygtool3/pygtool3/")

import os
import pygtool_core
import pygtool
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *
from plt import *

geogrid = pygtool.readgrid()
Clon, Clat = geogrid.getlonlat()

BASE_DIR = "/mnt/nj2/ngoc/CHASER_output"
CASES = [
    "VISITst20012023_CEDS",
    "UKpft20012023_CEDS",
    "MEGANpft20012023_CEDS",
    "BVOCoff20012023_CEDS",
]

# CASES = [
# "VISITst20012023_nudg",
# "UKpft20012023_nudg",
# "MEGANst20012023_nudg",
# "MEGANpft20012023_nudg",
# "UKst20012023_nudg",
# "MIXpft20012023_nudg",
# "BVOCoff20012023_nudg",
# ]

SIGMA = load_sigma()

list_regions = [
    "AMZ",
    "ENA",
    "NAU",
    # "SAF",
    # "MED",
    # "CEU",
    # "EAS",
    # "SAS",
    # "SEA",
    # "REMOTE_PACIFIC",
    # "Amazonia",
    # "S-E US",
    "Mato Grosso",
    "Indonesia",
    "South China",
    "C_Africa",
    "N_Africa",
    "S_Africa",
    # "WAF",
    # "SSA",
    # "NAS",
]

colors = [
    "#D55E00",  # vermillion
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    # "#CC79A7",  # reddish purple
    "red",
    # "#999933",  # olive green
    "#882255",
    "#AA4499",  # deep magenta
    "#44AA99",  # teal green
    "#332288",  # dark navy
]

ylim_dict = {
    "AMZ": (0, 20),
    "ENA": (0, 15),
    # "SAF": (0, 15),
    # "MED": (0, 15),
    # "CEU": (0, 15),
    "EAS": (0, 15),
    # "SAS": (0, 15),
    "SEA": (0, 20),
    "NAU": (0, 15),
    "Indonesia": (0, 20),
    "C_Africa": (0, 20),
    "N_Africa": (0, 20),
    "S_Africa": (0, 20),
    # "REMOTE_PACIFIC": (0, 4),
}


def sampling_12to14h_hoque():
    def process_ps(ds):
        ps = ds["PS"].values
        ps_layers = [ps * SIGMA[l] for l in range(len(SIGMA))]
        ps_layers = np.stack(ps_layers, axis=-1)
        return xr.Dataset(
            {
                "ps": (
                    ("time", "lat", "lon", "layer"),
                    ps_layers,
                )
            },
            coords={
                "time": ds.time.values,
                "lat": ds.lat.values,
                "lon": ds.lon.values,
                "layer": np.arange(len(SIGMA)),
            },
        )

    vars = ["ch2o", "t", "ps"]
    years = np.arange(2019, 2021)
    # years = np.arange(2001, 2024)
    sigma = np.arange(1, 37)

    case_dir = "/mnt/dg2/hoque/TROPOMI_model_analysis"
    for var in vars:

        out_dir = f"{BASE_DIR}/sample_12to14h/hoque_sim"
        out_file = f"{out_dir}/{var}.nc"

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if not os.path.exists(out_file):

            list_ds = []
            for y in years:
                if var == "ch2o":
                    chaser_var_path = (
                        f"{case_dir}/New_emission_chaser_2019_2020/{y}/{var}"
                    )
                else:
                    chaser_var_path = f"{case_dir}/{y}/{var}"
                time = pd.date_range(
                    f"{y}-01-01 02:00", f"{y+1}-01-01 00:00", freq="2H"
                )
                if var != "ps":
                    gtool = pygtool_core.Gtool3d(chaser_var_path, count=len(time))
                    gtool.set_datetimeindex(time)
                    var_ds = gtool.to_xarray(
                        lat=Clat, lon=Clon, sigma=sigma, na_values=np.nan
                    )
                else:
                    gtool = pygtool_core.Gtool2d(chaser_var_path, count=len(time))
                    gtool.set_datetimeindex(time)
                    var_ds = gtool.to_xarray(lat=Clat, lon=Clon, na_values=np.nan)
                sampled_ds = var_ds.sel(time=(var_ds.time.dt.hour.isin([12, 14])))
                sampled_ds = sampled_ds.groupby(sampled_ds.time.dt.date).mean()
                sampled_ds = sampled_ds.rename({"date": "time"})
                sampled_ds = process_ps(sampled_ds) if var == "ps" else sampled_ds

                list_ds.append(sampled_ds)

            concat_ds = xr.concat(list_ds, dim="time")
            concat_ds["time"] = np.array(concat_ds.time.values, dtype="datetime64[D]")

            concat_ds.to_netcdf(out_file)
        print(var)


def hcho_adj_by_ps(t_ds, ps_ds, ch2o_ds):

    def adj(p, p_next, t, t_next, hcho):
        def p2h(p, t):
            return (-1) * ((np.log(p * 100 / 101325)) * 8.314 * t) / (0.028 * 9.8)

        H = abs(p2h(p, t) - p2h(p_next, t_next))
        return (hcho * 1e-9 * H * p * 100 * 1e-4) / (1.38e-23 * t)

    c_temp = t_ds["T"].values  # (time, lat, lon, layer)
    c_ch2o = ch2o_ds["CH2O"].values  # (time, lat, lon, layer)
    c_ps = ps_ds["PS"].values  # (time, lat, lon, layer)

    c_time = t_ds.time.values
    hcho_layers = []
    for l in range(15):
        p = c_ps[:, :, :, l]
        p_next = c_ps[:, :, :, l + 1]

        t = c_temp[:, :, :, l]
        t_next = c_temp[:, :, :, l + 1]
        hcho = c_ch2o[:, :, :, l]

        ak_hcho = adj(p, p_next, t, t_next, hcho)

        hcho_layers.append(ak_hcho)

    assert len(hcho_layers) == 15

    hcho_ds = xr.Dataset(
        {
            "hcho": (
                ("layer", "time", "lat", "lon"),
                np.stack(hcho_layers, axis=0),
            )
        },
        coords={
            "layer": np.arange(len(hcho_layers)),
            "time": c_time,
            "lat": t_ds.lat.values,
            "lon": t_ds.lon.values,
        },
    )

    return hcho_ds["hcho"].sum("layer", skipna=True)


def chaser_tcol_cal():

    out_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_no_ak"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for _case in CASES:
        print(_case)
        case_dir = f"{BASE_DIR}/sample_12to14h/{_case}"

        t_ds = xr.open_dataset(f"{case_dir}/t.nc").resample(time="M").mean()
        ch2o_ds = xr.open_dataset(f"{case_dir}/ch2o.nc").resample(time="M").mean()
        ps_ds = xr.open_dataset(f"{case_dir}/ps.nc").resample(time="M").mean()

        t_ds = prep_chaser_for_ak(t_ds)
        ch2o_ds = prep_chaser_for_ak(ch2o_ds)
        ps_ds = prep_chaser_for_ak(ps_ds)

        mtype = "datetime64[M]"
        nstype = "datetime64[ns]"

        t_ds["time"] = t_ds.time.values.astype(mtype).astype(nstype)
        ch2o_ds["time"] = ch2o_ds.time.values.astype(mtype).astype(nstype)
        ps_ds["time"] = ps_ds.time.values.astype(mtype).astype(nstype)

        ps_ds = ps_ds.rename({list(ps_ds.data_vars)[0]: "PS"})

        adj_hcho = hcho_adj_by_ps(t_ds, ps_ds, ch2o_ds)
        adj_hcho.to_netcdf(f"{out_dir}/{_case}_15layers.nc")


def load_hcho_data():
    out_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_no_ak"
    hcho = {c: HCHO(f"{out_dir}/{c}_15layers.nc") for c in CASES}

    # this var is before correction
    # hcho["TROPOMI"] = HCHO(TROPO_FILE, hcho_var="tcolhcho_apriori")
    # hcho["non-AKed_OMI"] = HCHO(OMI_FILE, hcho_var="tcolhcho_apriori")

    return hcho


def load_contri_data():
    bvoc_contri_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/bvoc_contri"

    cases = [
        "VISITst20012023_nudg",
        "UKpft20012023_nudg",
        "MEGANst20012023_nudg",
        "MEGANpft20012023_nudg",
        "UKst20012023_nudg",
        "MIXpft20012023_nudg",
    ]

    bvoc_contri = {c: HCHO(f"{bvoc_contri_dir}/{c}_bvoc_contri.nc") for c in cases}

    return bvoc_contri


def plt_reg(hcho, unit=None):
    tits = ["Seasonal", "Inter-annual Variability"]
    for k, mode in enumerate(["ss", "ann"]):
        index = "month" if mode == "ss" else "year"
        fig, axis = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3), layout="constrained")

        for i, r in enumerate(list_regions):
            ri, ci = i // 3, i % 3
            ax = axis[ri, ci]
            for j, c in enumerate(list(hcho.keys())):
                ds = hcho[c].reg_ss if mode == "ss" else hcho[c].reg_ann
                reg_df = (
                    ds[[index, r]]
                    .set_index(index)
                    .rename(columns={r: c.replace("20012023_nudg", "")})
                )

                sns.lineplot(
                    reg_df,
                    ax=ax,
                    palette=[colors[j]],
                    markers=True,
                    lw=2,
                )
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_xlabel("Year")
            if mode == "ss":
                ax.set_xlabel("Month")
                ax.set_xticks(np.arange(1, 13))
            if ri < 2:
                ax.set_xlabel("")
            if unit is None:
                unit = "(\u00d710$^{15}$ molec.cm$^{-2}$)"
            ax.set_ylabel(unit)
            ax.set_title(f"{r}")
            ax.set_xlim(2005, 2023)
            # if r in ylim_dict:
            #     ax.set_ylim(ylim_dict[r])

        fig.legend(
            handles,
            labels,
            ncol=3,
            loc="center",
            bbox_to_anchor=(0.5, -0.06),
        )
        plt.suptitle(tits[k], fontsize=16, fontweight="bold")


# differency between bvoc_on vs bvoc_off means
def plt_reg_bvoc_ch4_contri(hcho, unit=None):

    tits = ["Seasonal", "Inter-annual Variability"]
    off_c = CASES[0]
    for k, mode in enumerate(["ss", "ann"]):
        index = "month" if mode == "ss" else "year"
        fig, axis = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3), layout="constrained")

        for i, r in enumerate(list_regions):
            ri, ci = i // 3, i % 3
            ax = axis[ri, ci]
            for j, c in enumerate(list(hcho.keys())[1:]):
                ds = hcho[c].reg_ss if mode == "ss" else hcho[c].reg_ann
                reg_df = ds[[index, r]].set_index(index).rename(columns={r: c})
                # bvoc off start
                ds_bvoc_off = (
                    hcho[off_c].reg_ss if mode == "ss" else hcho[off_c].reg_ann
                )
                reg_bvoc_off = (
                    ds_bvoc_off[[index, r]].set_index(index).rename(columns={r: c})
                )
                if c not in ["OMI", "TROPOMI"]:
                    reg_df[c] = reg_df[c] - reg_bvoc_off[c]
                # bvoc off end
                sns.lineplot(
                    reg_df,
                    ax=ax,
                    palette=[colors[j]],
                    markers=True,
                    lw=2,
                )
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_xlabel("Year")
            if mode == "ss":
                ax.set_xlabel("Month")
                ax.set_xticks(np.arange(1, 13))
            if ri < 2:
                ax.set_xlabel("")
            if unit is None:
                unit = "(\u00d710$^{15}$ molec.cm$^{-2}$)"
            ax.set_ylabel(unit)
            ax.set_title(f"{r}")
            ax.set_xlim(2005, 2023)
            # if r in ylim_dict:
            #     ax.set_ylim(ylim_dict[r])

        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="center",
            bbox_to_anchor=(0.5, -0.06),
        )
        plt.suptitle(tits[k], fontsize=16, fontweight="bold")


def process_ch4(c):
    years = np.arange(2005, 2024)
    list_ds = []
    for y in years:
        ch4_path = f"{BASE_DIR}/{c}/{y}/ch4"
        time = pd.date_range(f"{y}-01-01", f"{y+1}-01-01", freq="M")
        gtool = pygtool_core.Gtool2d(ch4_path, count=len(time))
        gtool.set_datetimeindex(time)
        list_ds.append(gtool.to_xarray(lat=Clat, lon=Clon, na_values=np.nan))
    ds = xr.concat(list_ds, dim="time")
    ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)
    ds = ds.sortby("lon")
    return ds


def bvoc_contribution():
    hcho_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_no_ak"
    bvoc_contri_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/bvoc_contri"

    cases = [
        "VISITst20012023_nudg",
        "UKpft20012023_nudg",
        "MEGANst20012023_nudg",
        "MEGANpft20012023_nudg",
        "UKst20012023_nudg",
        "MIXpft20012023_nudg",
    ]

    bvoc_case_off = "BVOCoff20012023_nudg"
    hcho_bvoc_off = xr.open_dataset(f"{hcho_dir}/{bvoc_case_off}_15layers.nc")
    hcho_bvoc_off = hcho_bvoc_off.sel(
        time=(hcho_bvoc_off.time.dt.year >= 2005) & (hcho_bvoc_off.time.dt.year < 2024)
    )
    ch4_bvoc_off = process_ch4(bvoc_case_off)

    for c in cases:
        hcho = xr.open_dataset(f"{hcho_dir}/{c}_15layers.nc")
        hcho = hcho.sel(time=(hcho.time.dt.year >= 2005) & (hcho.time.dt.year < 2024))
        ch4 = process_ch4(c)

        assert len(hcho.time) == len(ch4.time)

        bvoc_contri = (hcho.hcho.values - hcho_bvoc_off.hcho.values) * (
            ch4.CH4.values / ch4_bvoc_off.CH4.values
        )
        bvoc_contri_ds = xr.Dataset(
            {
                "bvoc_contri": (
                    ("time", "lat", "lon"),
                    bvoc_contri,
                )
            },
            coords={
                "time": hcho.time.values,
                "lat": hcho.lat.values,
                "lon": hcho.lon.values,
            },
        )
        if not os.path.exists(bvoc_contri_dir):
            os.makedirs(bvoc_contri_dir)

        bvoc_contri_out_file = f"{bvoc_contri_dir}/{c}_bvoc_contri.nc"
        if os.path.exists(bvoc_contri_out_file):
            os.remove(bvoc_contri_out_file)
            print(f"Remove {bvoc_contri_out_file}")

        bvoc_contri_ds.to_netcdf(f"{bvoc_contri_dir}/{c}_bvoc_contri.nc")


class CH4(HCHO):
    def __init__(self, case_):
        self.ds = process_ch4(case_)

        if "hcho" not in list(self.ds.data_vars):
            self.ds = self.ds.rename({list(self.ds.data_vars.keys())[0]: "hcho"})
        self.ds = self.ds.fillna(0)
        self.hcho = self.ds["hcho"]

        self.mask_land()

        self.cal_weights()
        (
            self.glob_ann,
            self.reg_ann,
            self.reg_ss,
            self.lat_mean,
        ) = HCHO.ann_ss_reg(self.hcho, self.weights)


def plt_ch4():
    ch4_obj = {c: CH4(c) for c in CASES}
    plt_reg(ch4_obj, unit="ppm")

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(8, 6),
        layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    for i, c in enumerate(["VISITst20012023_nudg", "UKpft20012023_nudg"]):
        ch4_2022 = (
            ch4_obj[c].ds.sel(time=slice("2022-01-01", "2022-12-31")).mean("time").hcho
        )
        ch4_2021 = (
            ch4_obj[c].ds.sel(time=slice("2021-01-01", "2021-12-31")).mean("time").hcho
        )
        diff = ch4_2022 - ch4_2021
        im = diff.plot(
            ax=ax[i], vmin=-0.1, vmax=0.1, cmap="coolwarm", add_colorbar=False
        )
        ax[i].set_title(f"{c}")
        ax[i].coastlines()
        ax[i].add_feature(cfeature.BORDERS, linewidth=0.5)
        ax[i].add_feature(cfeature.LAND, facecolor="lightgray", zorder=-1)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.8, pad=0.05)
    cbar.set_label("CH4 difference 2022-2021 (ppm)")


# %%


def oh_adj_by_ps(t_ds, ps_ds, oh_ds):

    def adj(p, p_next, t, t_next, oh):
        def p2h(p, t):
            return (-1) * ((np.log(p * 100 / 101325)) * 8.314 * t) / (0.028 * 9.8)

        H = abs(p2h(p, t) - p2h(p_next, t_next))
        return oh * H * 1e-10 * 6.022e23

    c_temp = t_ds["T"].values  # (time, lat, lon, layer)
    c_oh = oh_ds["OH"].values  # (time, lat, lon, layer)
    c_ps = ps_ds["PS"].values  # (time, lat, lon, layer)

    c_time = t_ds.time.values
    oh_layers = []
    for l in range(15):
        p = c_ps[:, :, :, l]
        p_next = c_ps[:, :, :, l + 1]

        t = c_temp[:, :, :, l]
        t_next = c_temp[:, :, :, l + 1]
        oh = c_oh[:, :, :, l]

        ak_oh = adj(p, p_next, t, t_next, oh)

        oh_layers.append(ak_oh)

    assert len(oh_layers) == 15

    oh_ds = xr.Dataset(
        {
            "oh": (
                ("layer", "time", "lat", "lon"),
                np.stack(oh_layers, axis=0),
            )
        },
        coords={
            "layer": np.arange(len(oh_layers)),
            "time": c_time,
            "lat": t_ds.lat.values,
            "lon": t_ds.lon.values,
        },
    )

    return oh_ds["oh"].sum("layer", skipna=True)


def chaser_tcol_cal():

    out_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_no_ak"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for _case in CASES:
        print(_case)
        case_dir = f"{BASE_DIR}/sample_12to14h/{_case}"

        t_ds = xr.open_dataset(f"{case_dir}/t.nc").resample(time="M").mean()
        oh_ds = xr.open_dataset(f"{case_dir}/oh.nc").resample(time="M").mean()
        ps_ds = xr.open_dataset(f"{case_dir}/ps.nc").resample(time="M").mean()

        t_ds = prep_chaser_for_ak(t_ds)
        oh_ds = prep_chaser_for_ak(oh_ds)
        ps_ds = prep_chaser_for_ak(ps_ds)

        mtype = "datetime64[M]"
        nstype = "datetime64[ns]"

        t_ds["time"] = t_ds.time.values.astype(mtype).astype(nstype)
        oh_ds["time"] = oh_ds.time.values.astype(mtype).astype(nstype)
        ps_ds["time"] = ps_ds.time.values.astype(mtype).astype(nstype)

        ps_ds = ps_ds.rename({list(ps_ds.data_vars)[0]: "PS"})

        adj_oh = oh_adj_by_ps(t_ds, ps_ds, oh_ds)
        adj_oh.to_netcdf(f"{out_dir}/oh_{_case}.nc")


def load_data():
    out_dir = f"/mnt/dg3/ngoc/emiisop_co2inhi_als/data/hcho_no_ak"
    oh = {c: HCHO(f"{out_dir}/oh_{c}.nc") for c in CASES}

    # this var is before correction
    # hcho["TROPOMI"] = HCHO(TROPO_FILE, hcho_var="tcolhcho_apriori")
    # hcho["non-AKed_OMI"] = HCHO(OMI_FILE, hcho_var="tcolhcho_apriori")

    return oh


def plt_reg(oh):

    colors = [
        "#D55E00",  # vermillion
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#009E73",  # bluish green
        "#F0E442",  # yellow
        "#0072B2",  # blue
        # "#CC79A7",  # reddish purple
        "red",
        # "#999933",  # olive green
        "#882255",
        "#AA4499",  # deep magenta
        "#44AA99",  # teal green
        "#332288",  # dark navy
    ]

    tits = ["Seasonal", "Inter-annual Variability"]
    for k, mode in enumerate(["ss", "ann"]):
        index = "month" if mode == "ss" else "year"
        fig, axis = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3), layout="constrained")

        for i, r in enumerate(list_regions):
            ri, ci = i // 3, i % 3
            ax = axis[ri, ci]
            for j, c in enumerate(list(oh.keys())):
                ds = oh[c].reg_ss if mode == "ss" else oh[c].reg_ann
                reg_df = (
                    ds[[index, r]]
                    .set_index(index)
                    .rename(columns={r: c.replace("20012023_nudg", "")})
                )
                sns.lineplot(
                    reg_df,
                    ax=ax,
                    palette=[colors[j]],
                    markers=True,
                    lw=2,
                )
            handles, labels = ax.get_legend_handles_labels()
            ax.get_legend().remove()
            ax.set_xlabel("Year")
            if mode == "ss":
                ax.set_xlabel("Month")
                ax.set_xticks(np.arange(1, 13))
            if ri < 2:
                ax.set_xlabel("")
            ax.set_ylabel("(molec.cm$^{-2}$)")
            ax.set_title(f"{r}")
            ax.set_xlim(2005, 2023)

        fig.legend(
            handles,
            labels,
            ncol=3,
            loc="center",
            bbox_to_anchor=(0.5, -0.06),
        )
        plt.suptitle(tits[k], fontsize=16, fontweight="bold")


# %% Plot means, trends, IAV of OH
import pymannkendall as pymk
import pandas as pd

list_regions = [
    "Amazonia",
    "S-E US",
    "Mato Grosso",
    "Indonesia",
    "South China",
    "C_Africa",
    "N_Africa",
    "S_Africa",
    "NAU",
]

records = []

for c in list(oh.keys())[1:]:  # skip first key if it's OBS or similar
    try:
        reg_df = oh[c].reg_ann  # DataFrame with 'year' and region columns
        years = reg_df["year"].values

        for region in list_regions:
            if region not in reg_df.columns:
                print(f"Region {region} not found in model {c}")
                continue

            values = reg_df[region].values

            # Mean and IAV
            reg_mean = values.mean()
            reg_max = values.max()
            reg_min = values.min()
            iav_percent = ((reg_max - reg_min) / reg_mean) * 100

            # Mann-Kendall Trend
            trend = pymk.original_test(values, alpha=0.05)
            trend_slope_abs = trend.slope
            trend_slope_pct_per_year = (trend_slope_abs / reg_mean) * 100

            # Store result
            records.append(
                {
                    "Region": region,
                    "Model": c.replace("20012023_nudg", ""),
                    "Mean (OH)": round(reg_mean, 2),
                    "Trend (abs)": round(trend_slope_abs, 4),
                    "Trend (%/yr)": round(trend_slope_pct_per_year, 2),
                    "Significant": trend.h,
                    "IAV (%)": round(iav_percent, 2),
                }
            )

    except Exception as e:
        print(f"Error for model {c}: {e}")

# Create summary DataFrame
df_oh_stats = pd.DataFrame.from_records(records)
df_oh_stats = df_oh_stats.sort_values(by=["Region", "Model"]).reset_index(drop=True)
df_oh_stats.to_csv(
    "/mnt/dg3/ngoc/emiisop_co2inhi_als/regional_oh_trends_20250925.csv", index=False
)

# Optional: Display
print(df_oh_stats)
