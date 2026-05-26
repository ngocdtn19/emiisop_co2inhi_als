# %%
# Combined 2 subplots (share x) + right-side panel titles + italic species names in legend

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# -----------------------
# Models
# -----------------------
def wilkinson2009(Ca, Ismax, h, cstar):
    Ca = np.asarray(Ca, dtype=float)
    return Ismax * (1.0 - 1.0 / ((cstar / Ca) ** h + 1.0))


def ukesm(Ca, cast):
    Ca = np.asarray(Ca, dtype=float)
    return cast / Ca


# -----------------------
# Data
# -----------------------
species = [
    dict(
        key="L.formosana",
        color="#EE6677",
        marker="o",
        x_plot=[0, 150, 240, 400, 1000, 1500],
        y_plot=[0, 1.45, 1.38, 1.0, 0.58, 0.5],
        x_fit=[150, 240, 400, 1000, 1500],
        y_fit=[1.45, 1.38, 1.0, 0.58, 0.5],
    ),
    dict(
        key="M.indica",
        color="#CCBB44",
        marker="o",
        x_plot=[0, 240, 400, 1000, 1500],
        y_plot=[np.nan, 1.05, 1.0, 0.7, 0.57],
        x_fit=[240, 400, 1000, 1500],
        y_fit=[1.05, 1.0, 0.7, 0.57],
    ),
    dict(
        key="P.tremula x P.tremuloides",
        color="k",
        marker="o",
        x_plot=[0, 240, 400, 600, 1200, 1500],
        y_plot=[np.nan, 1.1, 1.0, 0.79, 0.4, 0.36],
        x_fit=[240, 400, 600, 1200, 1500],
        y_fit=[1.1, 1.0, 0.79, 0.4, 0.36],
    ),
    dict(
        key="Q.rubra",
        color="#4477AA",
        marker="o",
        x_plot=[0, 240, 400, 1000, 1500],
        y_plot=[np.nan, 0.95, 1.0, 0.87, 0.81],
        x_fit=[240, 400, 1000, 1500],
        y_fit=[0.95, 1.0, 0.87, 0.81],
    ),
    dict(
        key="A.nigrescens",
        color="#AA3377",
        marker="o",
        x_plot=[0, 180, 280, 370, 400, 600],
        y_plot=[np.nan, 1.970854, 1.367418, 1.071849, 1.0, 0.690692],
        x_fit=[180, 280, 370, 400, 600],
        y_fit=[1.970854, 1.367418, 1.071849, 1.0, 0.690692],
        lw_fit=3,
    ),
]


# -----------------------
# Helpers
# -----------------------
def _finite_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def italic_name(name: str) -> str:
    # Mathtext italics; keep spaces visible in mathtext
    return r"$\it{" + name.replace(" ", r"\ ") + "}$"


def plot_panel(ax, model_name, model_func, p0, bounds, legend_fmt):
    # Smooth curve range (avoid Ca=0)
    x_grid = np.linspace(150, 1500, 400)

    for sp in species:
        # measured
        x_m, y_m = _finite_xy(sp["x_plot"], sp["y_plot"])
        ax.plot(
            x_m,
            y_m,
            color=sp["color"],
            marker=sp.get("marker", "o"),
            linestyle="-",
            label=f"{italic_name(sp['key'])} (measured)",
        )

        # fit
        x_f, y_f = _finite_xy(sp["x_fit"], sp["y_fit"])
        popt, _ = curve_fit(
            model_func,
            x_f,
            y_f,
            p0=p0,
            bounds=bounds,
            maxfev=20000,
        )

        # fitted curve (smooth)
        y_grid = model_func(x_grid, *popt)

        ax.plot(
            x_grid,
            y_grid,
            color=sp["color"],
            linestyle="--",
            linewidth=sp.get("lw_fit", 1.5),
            label=legend_fmt(sp["key"], popt),
        )

    ax.set_ylim(0, 2.2)
    ax.set_xlim(0, 1800)
    ax.set_ylabel("I$_x$/I$_{400}$")

    # title on the right of the panel
    ax.text(
        1.15,
        0.92,
        model_name,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
    )

    # legend on the right
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.45))


# -----------------------
# Build combined figure (2 panels share x)
# -----------------------
fig, axes = plt.subplots(
    2,
    1,
    figsize=(9, 7),
    sharex=True,
    constrained_layout=True,
)

# Top panel: MEGAN-based (Wilkinson 2009)
plot_panel(
    axes[0],
    model_name="The MEGAN-based function",
    model_func=wilkinson2009,
    p0=(1.0, 1.0, 400.0),
    bounds=([0.0, 0.0, 1.0], [np.inf, np.inf, np.inf]),
    legend_fmt=lambda n, p: (
        f"{italic_name(n)} " f"(I$_{{smax}}$={p[0]:.3f}, h={p[1]:.4f}, C*={p[2]:.0f})"
    ),
)

# Bottom panel: UK-based
plot_panel(
    axes[1],
    model_name="The UK-based function",
    model_func=ukesm,
    p0=(400.0,),
    bounds=([1.0], [np.inf]),
    legend_fmt=lambda n, p: f"{italic_name(n)} (Ca$_{{st}}$={p[0]:.3f})",
)

# shared x label only on bottom
axes[1].set_xlabel("Ca (ppm)")
axes[0].set_xlabel("")

plt.show()
# %%
