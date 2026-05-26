import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib as mpl
import xarray as xr
import pandas as pd
import pickle

from scipy import stats
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from mypath import *
import mk
import pymannkendall as pymk

title_sz = 16
legend_sz = 14
unit_sz = 12

colors_dict = {
    "co2": "#9970ab",
    "co2f": "#762a83",
    "co2fi": "#9970ab",
    "lulcc": "#acd39e",
    "clim": "#EE99AA",
    "all": "#44bb99",
    "tas": "#e31a1c",
    "rsds": "#fee08b",
    "pr": "#386cb0",
}

# linestyles_dict = {
#     "co2": "-",
#     "co2f": "-",
#     "co2fi": "-",
#     "lulcc": "-",
#     "clim": "-.",
#     "all": "--",
#     "tas": "-",
#     "rsds": "-",
#     "pr": "-",
# }

linestyles_dict = {
    "co2": "--",
    "co2f": "--",
    "co2fi": "--",
    "lulcc": "--",
    "clim": "--",
    "all": "-",
    "tas": "--",
    "rsds": "--",
    "pr": "--",
}


def plt_glob_trends_by_driver(models, mode="main"):
    rows = 1
    cols = 3
    fig, axes = plt.subplots(
        rows,
        cols,
        sharey=True,
        figsize=(3 * cols, 4.5 * rows),
        layout="constrained",
    )
    for i, m in enumerate(models.keys()):
        # r = i // cols
        c = i % cols
        ax = axes[c]

        if mode == "main":
            df = models[m].main_df_rate
        else:
            df = models[m].clim_df_rate
        list_drivers = df["driver"].values
        color_list = [colors_dict[d] for d in list_drivers]
        barplot = sns.barplot(
            df,
            x="driver",
            y="slope",
            ax=ax,
            palette=sns.color_palette(color_list),
        )
        for p, sig in zip(barplot.patches, df["sig"]):
            if sig == True:
                h = p.get_height()
                add_h = 0.1
                if mode == "clim":
                    add_h = 0.1
                h = h if h > 0 else h - add_h
                barplot.text(p.get_x() + p.get_width() / 2.0, h, "*", ha="center")
                print(h)
        ax.set_title(m, fontsize=title_sz)
        ax.set_xlabel("")
        ax.set_ylabel("")
        # if r in [0, 1]:
        axes[0].set_ylabel("Isoprene emission trends [TgC yr$^{-2}$]", fontsize=unit_sz)

        ax.set_ylim(-2.2, 2.2)
        if mode == "clim":
            ax.set_ylim(-0.75, 0.75)
        # fig_name = "Fig7"
        # if mode == "clim":
        #     ax.set_ylim(-0.25, 0.25)
        #     fig_name = "Fig9"
        # if fig_name:
        #     path_ = f"../figures/{fig_name}.tiff"
        #     fig.savefig(
        #         path_,
        #         format="tiff",
        #         dpi=300,
        #         bbox_inches="tight",
        #     )


def plt_glob_changes_by_driver(models, mode="main"):

    rows = 1
    cols = 3
    fig, axes = plt.subplots(
        rows,
        cols,
        sharex=True,
        sharey=True,
        figsize=(4 * cols, 4 * rows),
        layout="constrained",
    )

    for i, m in enumerate(models.keys()):
        # r = i // cols
        c = i % cols
        ax = axes[c]
        axbox = ax.get_position()

        ax.axhline(0, color="black", linewidth=0.75, ls="--")

        if mode == "main":
            df = models[m].main_rates_ts
            legend_elements = [
                Line2D([0], [0], color="#762a83", lw=2, label="co$_2$f", ls="--"),
                Line2D([0], [0], color="#9970ab", lw=2, label="co$_2$fi", ls="--"),
                Line2D([0], [0], color="#acd39e", lw=2, label="lulcc", ls="--"),
                Line2D([0], [0], color="#ee99aa", ls="--", lw=2.5, label="clim"),
                Line2D([0], [0], color="#44bb99", ls="-", lw=2.5, label="all"),
            ]
        else:
            df = models[m].clim_rates_ts
        pred_fields = df.columns
        for f in pred_fields:
            obj = df[f]
            x, y = obj.index, obj.values
            ax.plot(
                x,
                y,
                label=f,
                linewidth=2,
                color=colors_dict[f],
                ls=linestyles_dict[f],
            )
        ax.set_ylim(-60, 60)
        # if mode == "clim":
        #     ax.set_ylim(-40, 40)
        # ax.set_ylim([-160, 110])
        ax.set_title(m, fontsize=title_sz)
        # if r in [0, 1]:
        axes[0].set_ylabel(
            "Isoprene emission changes [TgC yr$^{-1}$]", fontsize=unit_sz
        )

    if mode == "main":
        fig.legend(
            handles=legend_elements,
            ncol=5,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            fontsize=legend_sz,
        )
        fig_name = "Fig6"

    else:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            ncol=4,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            fontsize=legend_sz,
        )
        fig_name = "Fig8"
