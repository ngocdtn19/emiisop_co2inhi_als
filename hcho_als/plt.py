# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import numpy as np
import seaborn as sns
import regionmask
import cartopy.crs as ccrs
import pymannkendall as pymk
import calendar
import glob

from sklearn.metrics import mean_squared_error as mse
from scipy.stats import pearsonr
from pathlib import Path

from utils import *
from max_doas import *

# AKED_CHASER_HCHO SETTING
AKED_DIR = "/mnt/nj2/ngoc/emiisop_co2inhi_als/data/hcho_sat_ak_applied"

SAT_NAMES = ["TROPO", "OMI"]
# CASES = [
#     "VISITst20012023_nudg",
#     "UKpft20012023_nudg",
#     "MEGANst20012023_nudg",
#     "MEGANpft20012023_nudg",
#     "UKst20012023_nudg",
#     "MIXpft20012023_nudg",
#     "BVOCoff20012023_nudg",
#     "OBS",
# ]
# BASE_CASE = "VISITst20012023_nudg"
# OFF_CASE = "BVOCoff20012023_nudg"

CASES = [
    "VISITst20052023_CEDS",
    "UKpft20052023_CEDS",
    "MEGANpft20052023_CEDS",
    "BVOCoff20052023_CEDS",
    "OBS",
]
BASE_CASE = "VISITst"
OFF_CASE = "BVOCoff"

SAT_CASE = ["OMI", "TROPOMI"]

ROIS = [
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
    "#009E73",  # bluish green
    # "#D55E00",  # vermillion
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    # "#F0E442",  # yellow
    # "#0072B2",  # blue
    # "#CC79A7",  # reddish purple
    # "red",
    # "#999933",  # olive green
    # "#882255",
    "#222222",
    "#AA4499",  # deep magenta
    "#44AA99",  # teal green
    "#332288",  # dark navy
]


def list_all_files(base_dir, sat_ver, cases=CASES):
    print(f"Searching in {base_dir}")
    print(f"for {sat_ver}")
    all_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            file = os.path.join(root, file)
            for case in cases:
                if case in file and sat_ver in file:
                    all_files.append(file)
    return all_files


def load_hcho(sat_name, sat_ver, layer_used):

    files = list_all_files(AKED_DIR, sat_ver)
    print(files)

    obs = HCHO(get_sat_file(sat_name, sat_ver), sat_filter=None)

    ak_files = [f for f in files if sat_name in f]
    interp_files = [f for f in ak_files if "/sat_interp/" in f]

    hcho_interp = {
        Path(f)
        .parents[1]
        .name.replace("20052023_CEDS", ""): HCHO(
            f, sat_filter=sat_name, layer_used=layer_used
        )
        for f in interp_files
    }
    obs_case = "OMI"
    if sat_name == "tropo":
        obs_case = "TROPOMI"
    hcho_interp[obs_case] = obs

    return hcho_interp


def load_hcho_maxdoas(sat_name, layer_used):

    files = list_all_files(AKED_DIR, "v2")
    files_by_sat = [f for f in files if sat_name in f]
    hcho = {
        Path(f)
        .parents[1]
        .name: HCHO_maxdoas(f, MAX_DOAS_COORDS, layer_used=layer_used)
        .hcho_at_maxdoas
        for f in files_by_sat
    }
    # hcho[sat_name] = obs
    hcho["MAX_DOAS"] = read_all_maxdoas_csv()

    return hcho


def get_sat_case(list_cases):
    for c in list_cases:
        if "OMI" in c:
            return c
    return None


def plt_maxdoas(sat_name="omi", layer_used=15, norm=False, summer_only=False):
    colors = [
        "#CC79A7",  # reddish purple
        "#009E73",  # bluish green
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#222222",
    ]
    hcho = load_hcho_maxdoas(sat_name, layer_used)

    f_annual, ax_annual = plt.subplots(3, 1, figsize=(6, 8), layout="constrained")
    f_ss, ax_ss = plt.subplots(3, 1, figsize=(6, 8), layout="constrained")
    max_doas_col = "MAX_DOAS"

    df_maxdoas = hcho[max_doas_col]
    for i, c in enumerate(hcho.keys()):
        df_case = hcho[c]

        for j, station in enumerate(MAX_DOAS_COORDS.keys()):
            df_station = df_case[station]
            df_ss = df_station.groupby(df_station.index.month).mean()
            ss_hcho = df_ss.hcho

            df_maxdoas_station = df_maxdoas[station]
            df_maxdoas_ss = df_maxdoas_station.groupby(
                df_maxdoas_station.index.month
            ).mean()
            ss_hcho_maxdoas = df_maxdoas_ss.hcho

            # df_annual = df_station.groupby(df_station.index.year).mean()
            # if summer_only:
            #     df_annual = df_station[df_station.index.month.isin([6, 7, 8])]
            #     df_annual = df_annual.groupby(df_annual.index.year).mean()

            # ann_hcho = df_annual.hcho
            if norm:
                ss_hcho = min_max_normalize(ss_hcho)
                ss_hcho_maxdoas = min_max_normalize(ss_hcho_maxdoas)
                # ann_hcho = min_max_normalize(ann_hcho)
            if c != max_doas_col:
                common = ss_hcho.dropna().index.intersection(
                    ss_hcho_maxdoas.dropna().index
                )
                if len(common) > 1:
                    R_ss, _ = pearsonr(ss_hcho.loc[common], ss_hcho_maxdoas.loc[common])
                    RMSE_raw = np.sqrt(
                        mse(ss_hcho_maxdoas.loc[common], ss_hcho.loc[common])
                    )

                    mean_ref = ss_hcho_maxdoas.loc[common].mean()
                    RMSE_pct = (RMSE_raw / mean_ref) * 100 if mean_ref != 0 else np.nan

                    ax_ss[j].annotate(
                        rf"$\it{{r}}$={R_ss:.2f}, RMSE={RMSE_pct:.2f}%",
                        xy=(1.02, 0.9 - 0.1 * i),  # stagger down per dataset
                        xycoords="axes fraction",
                        va="top",
                        ha="left",
                        fontsize=10,
                        color=colors[i],  # match line color
                    )

            # --- Plot ---
            ax_ss[j].plot(
                df_ss.index, ss_hcho * 1e-16, label=f"{c}", color=colors[i], marker="o"
            )
            ax_ss[j].set_title(f"{station}")
            ax_ss[j].set_xticks(df_ss.index)

            ax_ss[j].set_xticklabels([calendar.month_abbr[m] for m in df_ss.index])
            unit = "(\u00d710$^{16}$ molec.cm$^{-2}$)"
            ax_ss[j].set_ylabel(unit)

            # ax_annual[j].plot(
            #     df_annual.index, ann_hcho, label=f"{c}", color=colors[i], marker="o"
            # )
            # ax_annual[j].set_title(f"{station} - (Annual Mean)")

            # ax_annual[j].set_xlim(2012, 2020)

    for fig in [f_ss, f_annual]:
        handles, labels = ax_ss[0].get_legend_handles_labels()
        labels = [lbl.replace("20052023_CEDS", "") for lbl in labels]

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=6,
            bbox_to_anchor=(0.5, -0.1),
            frameon=False,
        )


def plt_reg(hcho, norm=False, unit=None, sslat=False):

    interested_case = list(hcho.keys())
    interested_case = [c for c in interested_case if "BVOCoff" not in c]

    ylim_dict = {
        # "SAF": (0, 15),
        # "MED": (0, 15),
        # "CEU": (0, 15),
        # "SAS": (0, 15),
        "AMZ": (0, 25),
        "ENA": (0, 20),
        "EAS": (0, 20),
        "SEA": (0, 15),
        "NAU": (0, 15),
        "Indonesia": (0, 15),
        "C_Africa": (0, 25),
        "N_Africa": (0, 25),
        "S_Africa": (0, 25),
        "REMOTE_PACIFIC": (0, 4),
    }
    trend_dict = {
        "Region": [],
        "Model": [],
        "Decadal trend (% per decade)": [],
        "Significant": [],
        "Period": [],
    }
    for mode in ["ss", "ann"]:
        index = "month" if mode == "ss" else "year"

        fig, axis = plt.subplots(3, 3, figsize=(3 * 3, 3 * 3), layout="constrained")

        for i, r in enumerate(ROIS):
            ri, ci = i // 3, i % 3
            ax = axis[ri, ci]
            for j, c in enumerate(interested_case):
                df = hcho[c].reg_ss
                if mode == "ann":
                    df = hcho[c].reg_ann
                    if sslat:
                        df = hcho[c].reg_ann_sslat
                reg_df = df[[index, r]].set_index(index).rename(columns={r: c})

                periods = (
                    ["2005-2014", "2013-2022", "2005-2022"]
                    if len(reg_df) >= 10
                    else ["2018-2023"]
                )
                if mode == "ann":
                    for p in periods:
                        start_year, end_year = map(int, p.split("-"))
                        reg_period_df = reg_df.loc[
                            (reg_df.index >= start_year) & (reg_df.index <= end_year)
                        ]
                        # trend = pymk.original_test(reg_period_df[c], alpha=0.05)
                        trend_per_dec, sig = linear_trend_significance(reg_period_df[c])
                        trend_dict["Period"].append(p)
                        trend_dict["Region"].append(r)
                        trend_dict["Model"].append(c)
                        # trend_dict["Trend (%)"].append(
                        #     trend.slope * 100 / reg_period_df[c].mean()
                        # )
                        # trend_dict["Significant"].append(1 if trend.h else 0)
                        trend_dict["Decadal trend (% per decade)"].append(trend_per_dec)
                        trend_dict["Significant"].append(sig)
                if norm:
                    reg_df[c] = min_max_normalize(reg_df[c])
                sns.lineplot(reg_df, ax=ax, palette=[colors[j]], markers=True, lw=2)
            # Set y-axis limit
            # if r in ylim_dict:
            #     ax.set_ylim(ylim_dict[r])

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

        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="center",
            bbox_to_anchor=(0.5, -0.06),
        )

    # plot trend
    trend_df = pd.DataFrame.from_dict(trend_dict)
    for p in trend_df["Period"].unique():
        # Create a new figure for this period
        fig, ax = plt.subplots(figsize=(12, 6))
        df_p = trend_df[trend_df["Period"] == p]

        # Create the bar plot on the new axes

        barplot = sns.barplot(
            data=df_p,
            x="Region",
            y="Decadal trend (% per decade)",
            hue="Model",
            palette=colors,
            ax=ax,
        )

        for patch in barplot.patches:
            h = patch.get_height()
            # Find the matching row where Linear trend (%) per decade equals height and Significant is True
            matching_row = df_p[
                (df_p["Decadal trend (% per decade)"] == h)
                & (df_p["Significant"] == True)
            ]

            if not matching_row.empty:
                add_h = 1.5  # height to add the star above the bar
                h = h if h > 0 else h - add_h
                ax.text(patch.get_x() + patch.get_width() / 2.0, h, "*", ha="center")

        ax.set_title(f"HCHO decadal trend (% per decade) during({p})")
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Trend (% per decade)")
        ax.set_xlabel("Region")
        plt.setp(ax.xaxis.get_ticklabels(), rotation=30)

        # Move legend outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.show()
    return trend_df


# Plot regional contributions of BVOC emissions to HCHO
def plt_reg_bvoc_contri(hcho, norm=False, unit=None, sslat=False):
    colors2 = [
        "#009E73",  # bluish green
        "#009E73",  # bluish green
        "#E69F00",  # orange
        "#56B4E9",  # sky blue
        "#222222",
    ]
    list_regions = [
        "AMZ",
        "ENA",
        "NAU",
        "Mato Grosso",
        "Indonesia",
        "South China",
        "C_Africa",
        "N_Africa",
        "S_Africa",
    ]
    trend_dict = {
        "Region": [],
        "Model": [],
        "Decadal trend (% per decade)": [],
        "Significant": [],
        "Period": [],
    }
    interested_case = list(hcho.keys())

    for mode in ["ss", "ann"]:
        index = "month" if mode == "ss" else "year"
        fig, axis = plt.subplots(3, 3, figsize=(9, 9), layout="constrained")

        for i, r in enumerate(list_regions):
            ri, ci = i // 3, i % 3
            ax = axis[ri, ci]

            for j, case in enumerate(interested_case):
                if case == OFF_CASE:
                    continue
                # Select time period
                ds = (
                    hcho[case].reg_ss
                    if mode == "ss"
                    else (hcho[case].reg_ann_sslat if sslat else hcho[case].reg_ann)
                )
                reg_df = ds[[index, r]].set_index(index).rename(columns={r: case})

                # Subtract BVOCoff (if applicable)
                ds_off = (
                    hcho[OFF_CASE].reg_ss
                    if mode == "ss"
                    else (
                        hcho[OFF_CASE].reg_ann_sslat
                        if sslat
                        else hcho[OFF_CASE].reg_ann
                    )
                )
                reg_off = ds_off[[index, r]].set_index(index).rename(columns={r: case})
                if case not in SAT_CASE:
                    reg_df[case] = reg_df[case] - reg_off[case]

                periods = (
                    ["2005-2014", "2013-2022", "2005-2022"]
                    if len(reg_df) >= 10
                    else ["2018-2023"]
                )
                if mode == "ann":
                    for p in periods:
                        start_year, end_year = map(int, p.split("-"))
                        reg_period_df = reg_df.loc[
                            (reg_df.index >= start_year) & (reg_df.index <= end_year)
                        ]
                        # trend = pymk.original_test(reg_period_df[c], alpha=0.05)
                        trend_per_dec, sig = linear_trend_significance(
                            reg_period_df[case]
                        )
                        trend_dict["Period"].append(p)
                        trend_dict["Region"].append(r)
                        trend_dict["Model"].append(case)
                        # trend_dict["Trend (%)"].append(
                        #     trend.slope * 100 / reg_period_df[c].mean()
                        # )
                        # trend_dict["Significant"].append(1 if trend.h else 0)
                        trend_dict["Decadal trend (% per decade)"].append(trend_per_dec)
                        trend_dict["Significant"].append(sig)

                if norm:
                    reg_df[case] = min_max_normalize(reg_df[case])

                sns.lineplot(
                    x=reg_df.index,
                    y=reg_df[case],
                    ax=ax,
                    label=case.replace("20012023_CEDS", ""),
                    color=colors2[j],
                    lw=2,
                    marker="o",
                )

            # Axis settings
            if unit is None:
                unit = "(\u00d710$^{15}$ molec.cm$^{-2}$)"
            ax.set_ylabel(unit)
            ax.set_title(f"{r}")
            ax.get_legend().remove()

            if mode == "ss":
                ax.set_xlabel("Month")
                ax.set_xticks(np.arange(1, 13))
            else:
                ax.set_xlabel("Year")

            if ri < 2:
                ax.set_xlabel("")

            # Optional: set y-axis limits if needed
            # if r in ylim_dict:
            #     ax.set_ylim(ylim_dict[r])

        # Global legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="center",
            bbox_to_anchor=(0.5, -0.06),
        )
    # plot trend
    trend_df = pd.DataFrame.from_dict(trend_dict)
    for p in trend_df["Period"].unique():
        # Create a new figure for this period
        fig, ax = plt.subplots(figsize=(12, 6))
        df_p = trend_df[trend_df["Period"] == p]

        # Create the bar plot on the new axes

        barplot = sns.barplot(
            data=df_p,
            x="Region",
            y="Decadal trend (% per decade)",
            hue="Model",
            palette=colors,
            ax=ax,
        )

        for patch in barplot.patches:
            h = patch.get_height()
            # Find the matching row where Linear trend (%) per decade equals height and Significant is True
            matching_row = df_p[
                (df_p["Decadal trend (% per decade)"] == h)
                & (df_p["Significant"] == True)
            ]

            if not matching_row.empty:
                add_h = 1.5  # height to add the star above the bar
                h = h if h > 0 else h - add_h
                ax.text(patch.get_x() + patch.get_width() / 2.0, h, "*", ha="center")

        ax.set_title(
            f"HCHO (BVOC contribution) decadal trend (% per decade) during({p})"
        )
        ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Trend (% per decade)")
        ax.set_xlabel("Region")
        plt.setp(ax.xaxis.get_ticklabels(), rotation=30)

        # Move legend outside plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.show()
    return trend_df


def plt_metric_reg(hcho_dict, bvoc_contri=False):

    list_case = list(hcho_dict.keys())
    obs_c = get_sat_case(list_case)

    list_reg = ROIS
    R = {"model": [c for c in list_case if c not in [obs_c, OFF_CASE]]}
    rmse = {"model": [c for c in list_case if c not in [obs_c, OFF_CASE]]}
    trend = {"model": [c for c in list_case if c not in [OFF_CASE]]}

    for reg in list_reg:
        R[reg] = []
        rmse[reg] = []
        for c in list_case:
            if c not in [obs_c, OFF_CASE]:
                model_reg = hcho_dict[c].reg_ann[reg]
                obs_reg = hcho_dict[obs_c].reg_ann[reg]
                if bvoc_contri:
                    model_reg = model_reg - hcho_dict[OFF_CASE].reg_ann[reg]

                R[reg].append(pearsonr(model_reg, obs_reg)[0])
                rmse[reg].append(
                    np.sqrt(mse(obs_reg, model_reg)) / obs_reg.mean() * 100
                )

        trend[reg] = []
        trend[f"{reg}_sig"] = []
        for c in list_case:
            if c != OFF_CASE:
                hcho_reg = hcho_dict[c].reg_ann[reg]
                if bvoc_contri and c != obs_c:
                    hcho_reg = hcho_reg - hcho_dict[OFF_CASE].reg_ann[reg]

                trend_dict = pymk.original_test(hcho_reg, alpha=0.05)
                trend[reg].append(trend_dict.slope)
                trend[f"{reg}_sig"].append(0 if not trend_dict.h else 1)

    rmse = pd.DataFrame.from_dict(rmse).round(2)
    R = pd.DataFrame.from_dict(R).round(2)
    trend = pd.DataFrame.from_dict(trend).round(3)

    tits = ["Normalized RMSE (%)", "R", "Trend"]
    for i, ds in enumerate([rmse, R, trend]):
        ds["model"] = ds["model"].apply(lambda x: x.replace("20012023_nudg", ""))
        ds.set_index("model", inplace=True)
        ds.index.name = "Model"
        ds = ds[[col for col in ds.columns if "sig" not in col]]

        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(ds.T, annot=True, cmap="YlGnBu")
        plt.title(tits[i], fontsize=16)
        plt.xticks(rotation=15)

    return rmse, R, trend


def plt_mean_ann(hcho, sslat=False):

    cases = list(hcho.keys())
    obs_c = get_sat_case(cases)
    cases.insert(0, cases.pop())

    rows, cols = len(cases) // 2, 2
    fig, axis = plt.subplots(
        rows,
        cols,
        figsize=(8 * rows, 5 * cols),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )
    unit = "(\u00d710$^{15}$ molec.cm$^{-2}$)"

    # cases.remove(obs_c)
    cases.remove(OFF_CASE)
    # cases.insert(0, obs_c)
    # cases.insert(2, "VISITst20012023_nudg")

    print(cases)

    mappable_group1 = None  # for j < 2
    mappable_group2 = None  # for j > 1

    obs_ds = hcho[obs_c].hcho.mean("time", skipna=True)
    if sslat:
        obs_ds = hcho[obs_c].hcho_sslat.mean("year", skipna=True)

    for j, c in enumerate(cases):

        ds = hcho[c].hcho.mean("time", skipna=True)
        if sslat:
            ds = hcho[c].hcho_sslat.mean("year", skipna=True)

        if j > 1:
            ds = (ds - obs_ds) * 100 / obs_ds

        ri, ci = j // cols, j % cols

        ax = axis[ri, ci]

        im = ds.plot(
            ax=ax,
            cmap="Spectral_r" if j < 2 else "bwr",
            add_colorbar=False,
            levels=16 if j < 2 else np.arange(-80, 81, 5),
            extend="max" if j < 2 else "both",
            vmin=0 if j < 2 else -80,
            vmax=15 if j < 2 else 80,
        )
        if j < 2 and mappable_group1 is None:
            mappable_group1 = im
        if j > 1 and mappable_group2 is None:
            mappable_group2 = im

        replace_str = "" if j < 2 else " (OBS subtracted)"
        tit = c.replace("20012023_nudg", replace_str)
        ax.set_title(tit, fontsize=18)
        ax.coastlines()
        ax.set_extent([-179.5, 179.5, -80, 80], crs=ccrs.PlateCarree())
    # Turn off unused axes
    cbar_ax1 = fig.add_axes([0.25, 0.55, 0.5, 0.01])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(mappable_group1, cax=cbar_ax1, orientation="horizontal")
    cbar1.set_label(unit, fontsize=16)
    cbar1.ax.tick_params(labelsize=14)

    # Add colorbar for j > 1 (anomalies)
    cbar_ax2 = fig.add_axes([0.25, 0.05, 0.5, 0.01])
    cbar2 = fig.colorbar(mappable_group2, cax=cbar_ax2, orientation="horizontal")
    cbar2.set_label("% Difference from OBS", fontsize=16)
    cbar2.ax.tick_params(labelsize=14)


def plt_map(hcho_dict, score="corr", sslat=False):
    fig, axis = plt.subplots(
        1,
        3,
        figsize=(6 * 3, 4.5 * 1),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    interested_case = list(hcho_dict.keys())
    obs_c = get_sat_case(interested_case)
    interested_case = [c for c in interested_case if c not in [obs_c, "BVOCoff"]]
    obs_ds = hcho_dict[obs_c].hcho
    obs_ds = obs_ds.groupby(obs_ds.time.dt.year).mean("time")

    if sslat:
        obs_ds = hcho_dict[obs_c].hcho_sslat

    for j, c in enumerate(interested_case):
        if c not in [obs_c, "BVOCoff"]:
            ri, ci = j // 2, j % 2
            # ax = axis[ri, ci]
            ax = axis[j]
            add_colorbar = False

            model_ds = hcho_dict[c].hcho
            model_ds = model_ds.groupby(model_ds.time.dt.year).mean("time")
            if sslat:
                model_ds = hcho_dict[c].hcho_sslat

            if score == "corr":
                corr, sig = map_corr_by_time(model_ds, obs_ds)

                cb = corr.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap="RdBu_r",
                    vmin=-1,
                    vmax=1,
                    add_colorbar=add_colorbar,
                )

                # Stippling
                lat, lon = np.meshgrid(corr["lat"], corr["lon"], indexing="ij")
                ax.plot(
                    lon[sig.values],
                    lat[sig.values],
                    "k.",
                    markersize=0.5,
                    transform=ccrs.PlateCarree(),
                )

            elif score == "rmse":
                rmse = compute_rmse(model_ds, obs_ds)

                cb = rmse.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap="rainbow",
                    vmin=0,
                    vmax=100,
                    add_colorbar=add_colorbar,
                )

            # Colorbar
            ax.coastlines()
            ax.set_title(c.replace("20012023_nudg", ""), fontsize=18)
    # Add a single shared colorbar at bottom center
    label = "Pearson R" if score == "corr" else "Normalized RMSE (%)"
    cbar_ax = fig.add_axes([0.2, 0.03, 0.6, 0.01])  # [left, bottom, width, height]
    cbar = fig.colorbar(cb, cax=cbar_ax, orientation="horizontal", label=label)
    cbar.set_label(label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)


def plt_trend(method="linear"):
    # years = ["2005_2014", "2005_2022", "2013_2022", "2018_2023"]
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

    colors = [
        "#08306b",  # dark blue
        "#2171b5",  # medium blue
        "#41ab5d",  # green
        "#a1d99b",  # light green/yellow
        "#fecf33",  # yellow
        "#f57f17",  # orange
        "#b30000",  # strong red
    ]

    cmap_diff = mcolors.LinearSegmentedColormap.from_list("trend_diff_modC", colors)
    years = ["2005_2022"]
    dir_name = "hcho_linear_slope"
    band = "emiisop_slope_percentage"
    if method == "mk":
        dir_name = "hcho_mk"
        band = "tcolhcho_ann_trend_percentage"
    for year in years:

        mk_files = glob(f"./plt_data/{dir_name}/*{year}.nc")
        mk_files = [f for f in mk_files if "BVOCoff" not in f]

        interested_case = []
        for f in mk_files:
            if "CEDS" in f:
                interested_case.append(f.split("/")[-1].split("20052023_CEDS")[0])
            elif "BIRA" in f:
                interested_case.append("OMI")
            else:
                interested_case.append("TROPOMI")
        fig, axis = plt.subplots(
            3,
            2,
            figsize=(8 * 2, 5 * 3),
            layout="constrained",
            subplot_kw=dict(projection=ccrs.PlateCarree()),
        )
        cbs = []
        f_sat = None
        f_visit = None
        f_megan = None
        f_uk = None
        for f in mk_files:
            if "OMI" in f:
                f_sat = f
            if "VISITst" in f:
                f_visit = f
            if "MEGANpft" in f:
                f_megan = f
            if "UKpft" in f:
                f_uk = f

        ds_sat = xr.open_dataset(f_sat)
        ds_visit = xr.open_dataset(f_visit)
        ds_megan = xr.open_dataset(f_megan)
        ds_uk = xr.open_dataset(f_uk)

        trend_ds_sat = ds_sat[band] * 10  # convert to % per decade
        trend_ds_visit = ds_visit[band] * 10  # convert to % per decade
        trend_ds_megan = ds_megan[band] * 10  # convert to % per decade
        trend_ds_uk = ds_uk[band] * 10  # convert to % per decade

        visit_sat_diff = trend_ds_visit - trend_ds_sat
        megan_sat_diff = trend_ds_megan - trend_ds_sat
        uk_sat_diff = trend_ds_uk - trend_ds_sat

        megan_better = abs(megan_sat_diff) < abs(visit_sat_diff)
        uk_better = abs(uk_sat_diff) < abs(visit_sat_diff)

        megan_worse = abs(megan_sat_diff) > abs(visit_sat_diff)
        uk_worse = abs(uk_sat_diff) > abs(visit_sat_diff)

        megan_diff = megan_worse + megan_better * 2
        megan_diff = megan_diff.where(megan_diff != 0)
        uk_diff = uk_worse + uk_better * 2
        uk_diff = uk_diff.where(uk_diff != 0)

        for j, f in enumerate(mk_files):
            c = interested_case[j]
            if "BVOCoff" not in c:

                ds = xr.open_dataset(f)
                trend_ds = ds[band] * 10  # convert to % per decade
                # if "VISITst" in c:
                #     trend_ds = trend_ds - trend_ds_sat

                if "OMI" not in c and "VISITst" not in c:
                    trend_ds = trend_ds - trend_ds_visit

                sig_mask = None
                if "emiisop_pval" in ds:
                    pval = ds["emiisop_pval"]
                    sig_mask = pval < 0.05

                ri, ci = j // 2, j % 2
                ax = axis[ri, ci]

                cb = trend_ds.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap="coolwarm" if ri == 0 else cmap_diff,
                    vmin=-12 if ri == 0 else -1,
                    vmax=12 if ri == 0 else 1,
                    levels=13 if ri == 0 else 9,
                    add_colorbar=False,
                )
                cbs.append(cb)

                # Add dots for significant pixels
                if sig_mask is not None:
                    lats = trend_ds["lat"].values
                    lons = trend_ds["lon"].values
                    yy, xx = np.meshgrid(lats, lons, indexing="ij")

                    ax.scatter(
                        xx[sig_mask],
                        yy[sig_mask],
                        color="k",
                        s=1,
                        transform=ccrs.PlateCarree(),
                        alpha=0.5,
                        zorder=3,
                    )
                # Colorbar
                ax.coastlines()
                ax.set_title(c, fontsize=24)
        diff_titles = ["MEGANpft", "UKpft"]
        for d, ds_diff in enumerate([megan_diff, uk_diff]):
            cmap_better = ListedColormap(["#d9d9d9", "#fb8072"])
            # cmap_better.set_bad("white")
            ax = axis[2, d]
            print(np.unique(ds_diff.values))
            cb_diff = ds_diff.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap_better,
                add_colorbar=False,
            )
            ax.set_title(diff_titles[d], fontsize=24)
            ax.coastlines()
        cbar = fig.colorbar(
            cb_diff, ax=axis[2, :], orientation="horizontal", shrink=0.2
        )

        bounds = [1, 1.5, 2]
        norm = mcolors.BoundaryNorm(bounds, cmap_better.N)
        cbar.set_ticks([1.25, 1.75])
        cbar.set_ticklabels(["Worse", "Better"])
        # cbar.set_label(
        #     "c) Regions where MEGANpft/UKpft improve over VISITst", fontsize=20
        # )
        cbar.ax.tick_params(labelsize=18)

        labels = [
            f"a) Decadal trend during {year} (% per decade)",
            f"b) Difference compared to VISITst during {year} (% per decade)",
        ]
        cbsl = [cbs[0], cbs[2]]

        for l in range(len(cbsl)):
            label = labels[l]
            cbar = fig.colorbar(
                cbsl[l],
                ax=axis[l, :],
                orientation="horizontal",
                label="",
                shrink=0.4,
            )
            # cbar.set_label(label, fontsize=20)
            cbar.ax.tick_params(labelsize=18)

        axes = axis.ravel() if hasattr(axis, "ravel") else [axis]

        for ax in axes:
            if not ax.has_data():  # Checks if anything was plotted
                ax.set_visible(False)


def plt_ann_glob(hcho):
    fig, axis = plt.subplots(1, 1, figsize=(4.5, 4), layout="constrained")
    dfs = [hcho.inhi_chaser_glob, hcho.tropomi_glob, hcho.no_inhi_chaser_glob]
    names = ["Inhi-CHASER", "TROPOMI", "No-Inhi-CHASER"]

    for i, df in enumerate(dfs):
        df = df.set_index("year")
        rename_df = df.rename(columns={"avg_glob_ann": names[i]})
        sns.lineplot(
            rename_df,
            ax=axis,
            palette=[colors[i]],
            markers=True,
        )
    axis.set_ylim(3.5, 5.5)
    axis.set_ylabel("(\u00d710$^{15}$ molec.cm$^{-2}$)")

    hcho_inhi_chaser = hcho.inhi_chaser_glob["avg_glob_ann"].values
    hcho_no_inhi_chaser = hcho.no_inhi_chaser_glob["avg_glob_ann"].values

    hcho_tropomi = hcho.tropomi_glob["avg_glob_ann"].values
    res_tropomi = pymk.original_test(hcho_tropomi, alpha=0.05)
    trend_tropomi = round(res_tropomi.slope, 2)
    trend_tropomi = trend_tropomi if not res_tropomi.h else f"{trend_tropomi}*"

    print(f"trend_tropomi:{trend_tropomi:.2f}")
    notes = ["inhi", "noinhi"]
    for i, hcho_chaser in enumerate([hcho_inhi_chaser, hcho_no_inhi_chaser]):

        res_chaser = pymk.original_test(hcho_chaser, alpha=0.05)

        trend_chaser = round(res_chaser.slope, 2)

        trend_chaser = trend_chaser if not res_chaser.h else f"{trend_chaser}*"

        # axis.text(2020, 3.7, trend_chaser, fontsize=12, color=colors[0])
        # axis.text(2021, 3.7, trend_tropomi, fontsize=12, color=colors[1])

        pearson_r, _ = pearsonr(hcho_chaser, hcho_tropomi)
        rmse = np.sqrt(mse(hcho_chaser, hcho_tropomi))

        axis.set_title(f" Global Mean Annual HCHO")
        print(notes[i])
        print(f"R:{pearson_r:.2f}")
        print(f"RMSE:{rmse:.2f}")
        print(f"trend chaser:{trend_chaser:.2f}")


def plt_ch4():
    obj = {"VISITst20012023_nudg": CH4()}
    plt_reg(obj, obj)


def plt_trend_isop_hcho(interp_omi_v2):
    name_dict = {
        "MEGANpft": "MEGANpft20012023_nudg",
        "MEGANst": "MEGANst20012023_nudg",
        "MIXpft": "MIXpft20012023_nudg",
        "UKpft": "UKpft20012023_nudg",
        "UKst": "UKst20012023_nudg",
        "woCO2inhi": "VISITst20012023_nudg",
    }
    emiisop_dir = "/mnt/dg3/ngoc/emiisop_co2inhi_als/data/processed_org_data/mk_trends_map/2005-2023/"
    rows, cols = 3, 2

    fig, axis = plt.subplots(
        rows,
        cols,
        figsize=(8 * cols, 4 * rows),
        layout="constrained",
        subplot_kw=dict(projection=ccrs.PlateCarree()),
    )

    lat_hcho = interp_omi_v2["MEGANpft20012023_nudg"].hcho.lat
    lon_hcho = interp_omi_v2["MEGANpft20012023_nudg"].hcho.lon

    for model in name_dict.keys():
        ax = axis.flatten()[list(name_dict.keys()).index(model)]
        emiisop_file = f"{emiisop_dir}/VISIT-wCO2inhi-{model}_emiisop.nc"
        if model == "woCO2inhi":
            emiisop_file = f"{emiisop_dir}/VISIT-woCO2inhi_emiisop.nc"
        emiisop_ds = xr.open_dataset(emiisop_file).emiisop
        emiisop_regrid = emiisop_ds.interp(lat=lat_hcho, lon=lon_hcho, method="nearest")

        hcho = interp_omi_v2[name_dict[model]].hcho_ann_mk.tcolhcho_ann_trend

        hcho_emiisop_ration = hcho / emiisop_regrid
        cb1 = hcho_emiisop_ration.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="RdBu_r",
            vmin=-0.01,
            vmax=0.01,
            add_colorbar=False,
        )
        ax.coastlines()
        ax.set_title(f"{model} hcho/emiisop trend ratio", fontsize=14)

        # Add a single shared colorbar at bottom center
        cbar_ax1 = fig.add_axes(
            [0.25, -0.02, 0.5, 0.01]
        )  # [left, bottom, width, height]
        cbar1 = fig.colorbar(cb1, cax=cbar_ax1, orientation="horizontal")
        cbar1.set_label("hcho/emiisop", fontsize=16)


def plt_regional_bbox():

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Rectangle

    # Define the regions
    regions = {
        "AMZ": {
            "lat": [11.439, -20.0],  # 11.439° to -20.0°
            "lon": [-79.729, -50.0],  # -79.729° to -50.0°
            "color": "darkgreen",
        },
        # AMZ: Amazon
        "ENA": {
            "lat": [50.0, 25.0],  # 50.0° to 25.0°
            "lon": [-85.0, -60.0],  # -85.0° to -60.0°
            "color": "navy",
        },
        # ENA: E. North America
        "NAU": {
            "lat": [-10.0, -30.0],  # -10.0° to -30.0°
            "lon": [110.0, 155.0],  # 110.0° to 155.0°
            "color": "coral",
        },
        "Mato Grosso": {
            "lat": [-10, -16],  # 10°S to 16°S
            "lon": [-60, -50],  # 60°W to 50°W
            "color": "red",
        },
        "Indonesia": {
            "lat": [6, -6],  # 6°N to 6°S (descending)
            "lon": [95, 112],  # 95°E to 112°E
            "color": "blue",
        },
        "South China": {
            "lat": [28, 22],  # 28°N to 22°N
            "lon": [100, 112],  # 100°E to 112°E
            "color": "green",
        },
        "C_Africa": {
            "lat": [6, -6],  # 6°N to 6°S
            "lon": [10, 35],  # 10°E to 35°E
            "color": "orange",
        },
        "N_Africa": {
            "lat": [15, 5],  # 15°N to 5°N
            "lon": [-10, 30],  # 10°W to 30°E → -10 to 30
            "color": "purple",
        },
        "S_Africa": {
            "lat": [-5, -15],  # 5°S to 15°S
            "lon": [10, 30],  # 10°E to 30°E
            "color": "brown",
        },
    }

    # Create the map
    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.OCEAN, color="lightblue", alpha=0.5)
    ax.add_feature(cfeature.LAND, color="lightgray", alpha=0.5)

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Plot each region as a rectangle
    for region_name, coords in regions.items():
        # Calculate rectangle parameters
        lat_min = min(coords["lat"])
        lat_max = max(coords["lat"])
        lon_min = min(coords["lon"])
        lon_max = max(coords["lon"])

        width = lon_max - lon_min
        height = lat_max - lat_min

        # Create rectangle
        rect = Rectangle(
            (lon_min, lat_min),
            width,
            height,
            linewidth=2,
            edgecolor=coords["color"],
            facecolor=coords["color"],
            alpha=0.3,
            transform=ccrs.PlateCarree(),
        )

        # Add rectangle to map
        ax.add_patch(rect)

        # Add label at center of rectangle
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2

        ax.text(
            center_lon,
            center_lat,
            region_name,
            transform=ccrs.PlateCarree(),
            ha="left",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Set global extent
    ax.set_global()

    # Add title
    plt.title(
        "Bounding box of regions of interest", fontsize=16, fontweight="bold", pad=20
    )

    # Add legend
    legend_elements = [
        patches.Patch(color=coords["color"], alpha=0.3, label=region_name)
        for region_name, coords in regions.items()
    ]
    plt.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0.02, 0.02))

    plt.tight_layout()
    plt.show()


# %%

interp_omi_v2 = load_hcho("omi", "v2", layer_used=14)
interp_tropo_v2 = load_hcho("tropo", "v2", layer_used=14)
# %%
