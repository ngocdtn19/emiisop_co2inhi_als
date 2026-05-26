# %%
import sys

sys.path.append("/home/ngoc/nc2gtool/pygtool3/pygtool3/")

import os
import pygtool_core
import pygtool
import xarray as xr
import pandas as pd
import numpy as np
import re
import joblib
import regionmask

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from glob import glob


def load_sigma():
    file = "/home/ngoc/nc2gtool/pygtool3/pygtool3/GTAXDIR/GTAXLOC.HETA36"
    data = open(file, "br")
    summ = str(int(36))
    dt = np.dtype(
        [
            ("f_header", ">i"),
            ("header", ">64S16"),
            ("1f_tail", "i"),
            ("2f_header", ">i"),
            ("arr", ">" + summ + "f"),
            ("2f_tail", ">i"),
        ]
    )
    heta = np.fromfile(data, dtype=dt)
    sigma = heta[0][4]
    return sigma


SIGMA = load_sigma()
geogrid = pygtool.readgrid()
Clon, Clat = geogrid.getlonlat()

CHASER_DIR = "/mnt/nj2/ngoc/CHASER_emission/"
VARS = ["ant_vocs", "bb_vocs", "bvoc_VISITst", "ch4+nox_emission"]
OMI_TIME = np.load("omi_time.npz")["arr"]


def prep_omi(fix_time=False):
    sat_dir = f"/mnt/dg3/ngoc/obs_data"
    time_omi = "20050101-20221231"
    m_name_omi = f"mon_BIRA_OMI_HCHO_L3_v2"

    omi_file = f"{sat_dir}/{m_name_omi}/EXTRACT/hcho_AERmon_{m_name_omi}_historical_gn_{time_omi}.nc"
    ds_omi = xr.open_dataset(omi_file)
    ds_omi_month = ds_omi.resample(time="MS").mean()
    ds_omi_month = ds_omi_month.dropna(dim="time", how="all")
    if fix_time:
        omi_time = ds_omi_month.time.values
        np.savez("omi_time.npz", arr=omi_time)
    return ds_omi_month.groupby("time.year").mean("time").tcolhcho


def read_gtool(path_, start, end, case_name, mode="2d", monthly=True):
    time = pd.date_range(f"{start}-01-01 00:00", f"{end}-12-01 00:00", freq="MS")
    if not monthly:
        time = pd.date_range(f"{start}-01-01 00:00", f"{end}-01-01 00:00", freq="YS")
    if mode == "3d":
        gtool = pygtool_core.Gtool3d(path_, count=len(time))
        gtool.set_datetimeindex(time)
        var_ds = gtool.to_xarray(lat=Clat, lon=Clon, sigma=SIGMA, na_values=np.nan)
    else:
        gtool = pygtool_core.Gtool2d(path_, count=len(time))
        gtool.set_datetimeindex(time)
        var_ds = gtool.to_xarray(lat=Clat, lon=Clon, na_values=np.nan)

    var_ds = var_ds.assign_coords(lon=((var_ds.lon + 180) % 360) - 180)
    var_ds = var_ds.sortby("lon")

    # filter omi time
    if monthly:
        var_ds = var_ds.sel(time=(time.isin(OMI_TIME)))
        assert len(var_ds.time.values) == len(
            OMI_TIME
        ), f"{len(var_ds.time.values)} vs {len(OMI_TIME)}"

    new_dir = os.path.dirname(
        path_.replace("CHASER_emission", f"CHASER_OMI_prepped/{case_name}")
    )
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    base_name = os.path.basename(path_)
    new_path = os.path.join(new_dir, f"{base_name}_omi.nc")

    var_ds = var_ds.groupby("time.year").mean("time")  # yearly mean
    assert len(var_ds.year) == 18, f"{len(var_ds.year)}"
    var_ds.to_netcdf(new_path)

    return var_ds


def load_chaser_data(case_name):
    all_gtool = glob(os.path.join(CHASER_DIR, "*", "*"))
    for path_ in all_gtool:
        s = os.path.basename(path_)
        if s != "CH4_conc-hRCP45.ppm":
            match = re.search(r"(\d+)-(\d+)", s)
            start, end = 1901, 2023
            monthly = True
            if match:
                start, end = match.groups()
        else:

            start, end = 2005, 2022
            monthly = False
            continue
        print(s, start, end)
        ds = read_gtool(path_, start, end, case_name, monthly=monthly)


def ann_ss_reg(ds, ws, sslat=False):

    hoque_regions = list(HCHO.regs.keys())
    list_regs = regionmask.defined_regions.srex.abbrevs + hoque_regions
    lls = ["lat", "lon"]

    mask_3D = regionmask.defined_regions.srex.mask_3D(ds.isel(time=0))
    years = np.unique(ds.time.dt.year.values)

    glob_hcho_ann = {"year": years, "avg_glob_ann": []}
    reg_hcho_ann = {**{r: [] for r in list_regs}, "year": years}

    # cal annual hcho for glob and reg
    for y in glob_hcho_ann["year"]:
        ds_y = ds.sel(time=(ds.time.dt.year == y))
        hcho_reg = {r: [] for r in list_regs}
        hcho_glob = []

        for m in range(1, 13):
            ds_month = ds_y.sel(time=(ds_y.time.dt.month == m))
            # cal global hcho
            hcho_m_w = ds_month.weighted(ws).mean(skipna=True).item()
            hcho_glob.append(hcho_m_w)
            # cal regional hcho
            for r in list_regs:
                if r not in hoque_regions:
                    mask_r = mask_3D.isel(region=(mask_3D.abbrevs == r))
                    hcho_m_w_r = ds_month.weighted(mask_r * ws).mean(skipna=True).item()
                else:
                    lat, lon = (
                        HCHO.regs[r]["lat"],
                        HCHO.regs[r]["lon"],
                    )
                    hcho_m_w_r = (
                        ds_month.sel(lat=slice(*lat), lon=slice(*lon))
                        .mean(skipna=True)
                        .item()
                    )
                hcho_reg[r].append(hcho_m_w_r)

        for r in list_regs:
            data_r = np.array(hcho_reg[r])
            valid_r = data_r[(data_r != 0) & ~np.isnan(data_r)]
            reg_hcho_ann[r].append(np.mean(valid_r))

        data_glob = np.array(hcho_glob)
        valid_glob = data_glob[(data_glob != 0) & ~np.isnan(data_glob)]
        glob_hcho_ann["avg_glob_ann"].append(np.mean(valid_glob))

    glob_hcho_ann = pd.DataFrame.from_dict(glob_hcho_ann)
    reg_hcho_ann = pd.DataFrame.from_dict(reg_hcho_ann)

    return glob_hcho_ann, reg_hcho_ann


class ChaserOmi:
    var_groups = {
        "ant_vocs": [
            "QFLXAROM",
            "QFLXC2H4",
            "QFLXC2H6",
            "QFLXC3H6",
            "QFLXC3H8",
            "QFLXCH3COCH3",
            "QFLXCH3OH",
            "QFLXONMV",
        ],
        "bb_vocs": [
            "BBFLXONMV_x",
            "BBFLXC2H4",
            "BBFLXC2H6",
            "BBFLXC3H6",
            "BBFLXC3H8",
            "BBFLXCH3COCH3",
            "BBFLXCH3OH",
            "BBFLXONMV_y",
        ],
        "bvoc_VISITst": [
            "BFLXC10H16",
            "BFLXC5H8",
            "BFLXCH3COCH3",
            "BFLXCH3OH",
            "BFLXONMV",
        ],
        "ch4nox_ems": ["QFLXCH4", "AIRNOXRCP", "QFLXNOX"],
    }

    def __init__(self, case_name):
        self.ds_omi = prep_omi()
        self.org_omi_name = "tcolhcho"
        self.new_omi_name = "omi_hcho"
        self.base_dir = f"/mnt/nj2/ngoc/CHASER_OMI_prepped/{case_name}"

        self.model_path = "rf_minmax_model.joblib"

        # simulations settings
        self.list_sims = [
            "all_input_vary",
            "bvoc_off",
            "fixed_ant",
            "fixed_bb",
            "fixed_ch4nox",
        ]
        self.list_estims = [
            "bvoc_contri",
            "ant_voc_contri",
            "bb_voc_contri",
            "ch4nox_contri",
        ]
        self.fixed_year = 2005

        self.load_data()
        self.extract_features()
        self.load_model()
        self.run_simulation()

    def load_data(self):
        self.all_files = glob(os.path.join(self.base_dir, "*", "*.nc"))
        merge_df = (
            self.ds_omi.to_dataframe()
            .reset_index()
            .rename(columns={self.org_omi_name: self.new_omi_name})
        )
        for f in self.all_files:
            var_ds = xr.open_dataset(f)
            var_df = var_ds.to_dataframe().reset_index()
            merge_df = merge_df.merge(
                var_df,
                on=["year", "lat", "lon"],
                how="inner",
            )
        self.merge_df = merge_df

    def extract_features(self):
        df = self.merge_df.fillna(0)

        self.X = df.drop(columns=[self.new_omi_name])
        self.y = df[self.new_omi_name]
        self.feature_cols = self.X.columns.tolist()
        self.target_col = self.new_omi_name

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Model already exists at: {self.model_path}")
            self.model = joblib.load(self.model_path)["model"]
        else:
            self.model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler()),
                    (
                        "rf",
                        RandomForestRegressor(n_estimators=500, n_jobs=-1),
                    ),
                ]
            )
            print("Training model...")
            self.model.fit(self.X, self.y)

            joblib.dump({"model": self.model}, self.model_path)
            print(f"Model saved to: {self.model_path}")

    def get_fixed_predictors(self, sim_id, year):
        if sim_id == "all_input_vary":
            fixed_cols = []
        elif sim_id == "bvoc_off":
            fixed_cols = self.var_groups["bvoc_VISITst"]
        elif sim_id == "fixed_ant":
            fixed_cols = self.var_groups["ant_vocs"]
        elif sim_id == "fixed_bb":
            fixed_cols = self.var_groups["bb_vocs"]
        elif sim_id == "fixed_ch4nox":
            fixed_cols = self.var_groups["ch4nox_ems"]

        keys = ["lat", "lon"]

        if sim_id == "bvoc_off":
            df_fixed = self.merge_df.copy()
            df_fixed[fixed_cols] = 0
            return df_fixed
        elif len(fixed_cols) > 0:
            df_fixed = self.merge_df[self.merge_df["year"] == year][fixed_cols + keys]
            df_no_fixed = self.merge_df.drop(fixed_cols, axis=1)
            df_fixed = df_no_fixed.merge(df_fixed, how="left", on=keys)
            return df_fixed
        return self.merge_df

    def run_simulation(self):
        """
        all_changes
        bvoc_off => bvoc = 0
        bvoc_contri = all - bvoc_off

        ant_voc => fix at 2005 level
        bb_voc => fix at 2005 level
        ch4+nox_emission => fix at 2005 level

        ant_voc_contri = all - ant_voc_fix
        bb_voc_contri = all - bb_voc_fix
        ch4+nox_contri = all - ch4+nox_fix

        """

        for sim_id in self.list_sims:
            print(f"Running simulation: {sim_id}")
            df_sim = self.get_fixed_predictors(sim_id, self.fixed_year)
            X_sim = df_sim[self.feature_cols].fillna(0)
            y_pred = self.model.predict(X_sim)
            self.merge_df[f"{sim_id}"] = y_pred

        self.merge_df["bvoc_contri"] = (
            self.merge_df["all_input_vary"] - self.merge_df["bvoc_off"]
        )
        self.merge_df["ant_voc_contri"] = (
            self.merge_df["all_input_vary"] - self.merge_df["fixed_ant"]
        )
        self.merge_df["bb_voc_contri"] = (
            self.merge_df["all_input_vary"] - self.merge_df["fixed_bb"]
        )
        self.merge_df["ch4nox_contri"] = (
            self.merge_df["all_input_vary"] - self.merge_df["fixed_ch4nox"]
        )
        self.sens_ds = self.merge_df.groupby(["year", "lat", "lon"]).mean().to_xarray()
        self.df_plot = (
            self.merge_df.dropna(subset=self.new_omi_name).groupby("year").mean()
        )
        self.df_plot[self.list_sims + [self.new_omi_name]].plot.line()

    def cross_val(self):
        # score = {
        #     "rmse_mean": 781521158642783.6,
        #     "rmse_std": 5233412307260.5205,
        #     "mae_mean": 570096974626191.0,
        #     "mae_std": 619282537754.8047,
        #     "r2_mean": 0.9088155069775637,
        #     "r2_std": 0.0007431219516970338,
        # }
        n_splits = 3
        random_state = 42

        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", MinMaxScaler()),
                (
                    "rf",
                    RandomForestRegressor(n_estimators=500, random_state=random_state),
                ),
            ]
        )

        # --- CV scheme ---
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        scoring = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

        out = cross_validate(
            model,
            self.X,
            self.y,
            cv=cv,
            scoring=scoring,
            return_estimator=False,
            n_jobs=-1,
        )

        results = {
            "rmse_mean": (-out["test_rmse"]).mean(),
            "rmse_std": (-out["test_rmse"]).std(ddof=1),
            "mae_mean": (-out["test_mae"]).mean(),
            "mae_std": (-out["test_mae"]).std(ddof=1),
            "r2_mean": out["test_r2"].mean(),
            "r2_std": out["test_r2"].std(ddof=1),
        }
        print(results)


# %%
