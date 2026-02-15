"""
Melbourne House Price Estimation Dashboard (Improved Model)
----------------------------------------------------------
- Loads dataset from GitHub RAW
- Mentions Kaggle as dataset origin (footer)
- Improved estimator: Ridge Regression with log(Price) + one-hot for Type and Suburb
- DistanceRange bins: 3km bins, right=False, sorted by Regionname then Distance (as notebook)
- Distance input constrained to the selected DistanceRange bounds (auto-sets to min on change)
- Scatter: Price vs Landsize, colored by DistanceRange
  * Initially shows only selected DistanceRange, others set to legendonly (user can toggle via legend)
- Auto-opens browser on run
"""

from __future__ import annotations

import math
import threading
import webbrowser
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dash import Dash, Input, Output, State, dcc, html
import dash_bootstrap_components as dbc


# ============================================================
# CONFIG
# ============================================================
DATASET_URL = (
    "https://raw.githubusercontent.com/JoseChirif/"
    "2018-House-Price-Estimation---Melbourne-Australia/refs/heads/main/"
    "Data/Melbourne_housing_FULL.csv"
)

KAGGLE_URL = (
    "https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market"
    "?select=Melbourne_housing_FULL.csv"
)

NUM_FEATURES = ["Rooms", "Bathroom", "Car", "Distance", "Landsize"]
CAT_FEATURES = ["Type", "Suburb"]

IQR_K = 1.5
LANDSIZE_CAP = 5000  # keeps visuals/model stable


# ============================================================
# UTILITIES
# ============================================================
def remove_outliers_iqr(df: pd.DataFrame, column: str, k: float = IQR_K) -> pd.DataFrame:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return df
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return df.loc[(df[column] >= lo) & (df[column] <= hi)].copy()


def add_distance_range_3km(df: pd.DataFrame) -> pd.DataFrame:
    distances = df["Distance"].dropna()
    if distances.empty:
        df["DistanceRange"] = np.nan
        return df

    bins_range = range(math.floor(distances.min()), int(df["Distance"].max()) + 3, 3)
    labels_range = [f"{i}-{i+3}" for i in list(bins_range)[:-1]]

    df["DistanceRange"] = pd.cut(
        df["Distance"],
        bins=bins_range,
        labels=labels_range,
        right=False,
    )

    df = df.sort_values(by=["Regionname", "Distance"]).reset_index(drop=True)
    return df


def parse_distance_range(lbl: str) -> Optional[Tuple[float, float]]:
    try:
        a, b = str(lbl).split("-")
        return float(a), float(b)
    except Exception:
        return None


def format_money_aud(x: float) -> str:
    if not np.isfinite(x):
        return "AUD —"
    if x >= 1_000_000:
        return f"AUD {x/1_000_000:,.2f}M"
    return f"AUD {x:,.0f}"


def safe_int(v, default: int) -> int:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return int(v)
    except Exception:
        return default


def safe_float(v, default: float) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def card(title: str, body) -> dbc.Card:
    return dbc.Card(
        [dbc.CardHeader(html.Div(title, style={"fontWeight": "600"})), dbc.CardBody(body)],
        className="shadow-sm",
    )


# ============================================================
# LOAD & PREPARE
# ============================================================
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATASET_URL).copy()

    for col in ["Price", "Landsize", "Distance", "Rooms", "Bathroom", "Car"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["Regionname", "Price"] + NUM_FEATURES
    df = df.dropna(subset=required).copy()
    df["Type"] = df["Type"].astype(str)
    df["Suburb"] = df["Suburb"].astype(str)

    df = df.loc[(df["Price"] > 0) & (df["Landsize"] > 0)].copy()
    df = df.loc[df["Landsize"] <= LANDSIZE_CAP].copy()

    if {"Regionname", "Suburb", "Address"}.issubset(df.columns):
        df = df.loc[~df.duplicated(subset=["Regionname", "Suburb", "Address"])].copy()

    df = add_distance_range_3km(df)
    df = df.dropna(subset=["DistanceRange"]).copy()

    # Work comprises from 9km away -> remove 0–9 km bins
    df = df.loc[~df["DistanceRange"].astype(str).isin({"0-3", "3-6", "6-9"})].copy()

    # IQR cleanup (keeps visuals sane; model also benefits)
    df = remove_outliers_iqr(df, "Price", k=IQR_K)
    df = remove_outliers_iqr(df, "Landsize", k=IQR_K)
    df = remove_outliers_iqr(df, "Distance", k=IQR_K)

    df["LogPrice"] = np.log(df["Price"])
    return df.reset_index(drop=True)


DF = load_data()


# ============================================================
# MODEL: Ridge + OneHot(Type, Suburb) + log target
# ============================================================
TYPE_LABELS = {
    "h": "House",
    "t": "Townhouse",
    "u": "Unit / Apartment",
}

def format_type_option(t: str) -> str:
    t0 = str(t).strip().lower()
    if t0 in TYPE_LABELS:
        return f"{t0} — {TYPE_LABELS[t0]}"
    return str(t)
def build_model() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_FEATURES),
        ],
        remainder="drop",
    )

    model = Ridge(alpha=1.0, random_state=42)

    return Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )


FEATURES_ALL = NUM_FEATURES + CAT_FEATURES
X = DF[FEATURES_ALL].copy()
y_log = DF["LogPrice"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

PIPE = build_model()
PIPE.fit(X_train, y_train)

yhat_test = PIPE.predict(X_test)
R2_LOG_TEST = float(r2_score(y_test, yhat_test))

# Also compute an interpretable MAE in original scale (approx):
y_test_price = np.exp(y_test.to_numpy())
yhat_test_price = np.exp(yhat_test)
MAE_AUD_TEST = float(mean_absolute_error(y_test_price, yhat_test_price))


# ============================================================
# APP
# ============================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

REGIONS = sorted([str(r) for r in DF["Regionname"].dropna().unique().tolist()])
DEFAULT_REGION = REGIONS[0] if REGIONS else ""


def distance_ranges_for_region(region: str) -> List[str]:
    d = DF.loc[DF["Regionname"] == region, "DistanceRange"].astype(str).unique().tolist()
    d = [x for x in d if x and x != "nan"]
    return sorted(d, key=lambda s: float(s.split("-")[0]) if "-" in s else 1e9)


DEFAULT_DISTANCE_RANGE = distance_ranges_for_region(DEFAULT_REGION)[0] if DEFAULT_REGION else ""


app.layout = dbc.Container(
    fluid=True,
    style={"padding": "16px"},
    children=[
        html.H2("Trained Predictive Systems — Melbourne House Price Estimation 2018"),
        html.Hr(),

        dbc.Row(
            [
                dbc.Col(
                    [
                        card(
                            "Filters & Inputs (for estimation)",
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Region"),
                                            dcc.Dropdown(
                                                id="region_dd",
                                                options=[{"label": r, "value": r} for r in REGIONS],
                                                value=DEFAULT_REGION,
                                                clearable=False,
                                            ),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Distance Range (km)"),
                                            dcc.Dropdown(id="distance_range_dd", clearable=False),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [dbc.Label("Type"), dcc.Dropdown(id="type_dd", clearable=False)],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [dbc.Label("Suburb"), dcc.Dropdown(id="suburb_dd", clearable=False)],
                                        md=6,
                                    ),
                                    dbc.Col([dbc.Label("Rooms"), dbc.Input(id="rooms_in", type="number", min=0, step=1, value=3)], md=4),
                                    dbc.Col([dbc.Label("Bathroom"), dbc.Input(id="bath_in", type="number", min=0, step=1, value=2)], md=4),
                                    dbc.Col([dbc.Label("Car"), dbc.Input(id="car_in", type="number", min=0, step=1, value=1)], md=4),

                                    dbc.Col(
                                        [
                                            dbc.Label("Distance (km)"),
                                            dbc.Input(id="distance_in", type="number", min=0, step=0.1, value=12.0),
                                            dbc.FormText("Constrained to the selected Distance Range."),
                                        ],
                                        md=6,
                                    ),
                                    dbc.Col(
                                        [dbc.Label("Landsize (m²)"), dbc.Input(id="landsize_in", type="number", min=1, step=1, value=500)],
                                        md=6,
                                    ),
                                ],
                                className="g-2",
                            ),
                        ),
                        html.Div(style={"height": "12px"}),
                        card(
                            "Exploratory Scatter (Price vs Landsize) — DistanceRange toggles via legend",
                            dcc.Graph(id="scatter_graph", config={"displayModeBar": True}),
                        ),
                    ],
                    md=6,
                ),

                dbc.Col(
                    [
                        card("Estimation Results", html.Div(id="results_box")),
                        html.Div(style={"height": "12px"}),
                        card("Price Distribution by Region", dcc.Graph(id="region_box_graph", config={"displayModeBar": True})),
                    ],
                    md=6,
                ),
            ],
            className="g-3",
        ),

        html.Hr(style={"marginTop": "18px"}),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5("Dataset origin"),
                        html.Div("The present work is based on the dataset of the following Kaggle's link:"),
                        dcc.Markdown(f"- {KAGGLE_URL}"),
                        html.Div("By Tony Pino"),
                    ],
                    md=6,
                ),
                dbc.Col(
                    [
                        html.H5("Limitations"),
                        html.Ul(
                            [
                                html.Li("The dataset is trained and uses only 2018 data, so it is important to consider that prices change over time."),
                                html.Li("Ridge Regression with one-hot encoding for Type and Suburb, using log(Price) target."),
                                html.Li("Outliers are reduced using IQR filtering for readability."),
                                html.Li(
                                    "Machine Learning work will comprise from 9km away from the City Centre in "
                                    "Western Victoria, Eastern Victoria and Northern Metropolitan regions."
                                ),
                            ]
                        ),
                    ],
                    md=6,
                ),
            ],
            className="g-3",
        ),
        html.Hr(),
        html.Div("© Jose Chirif - www.josechirif.com", style={"textAlign": "center"}),
    ],
)


# ============================================================
# CALLBACKS
# ============================================================
@app.callback(
    Output("distance_range_dd", "options"),
    Output("distance_range_dd", "value"),
    Input("region_dd", "value"),
)
def update_distance_range_dd(region: str):
    region = str(region) if region else DEFAULT_REGION
    opts = distance_ranges_for_region(region)
    val = opts[0] if opts else None
    return [{"label": o, "value": o} for o in opts], val


@app.callback(
    Output("type_dd", "options"),
    Output("type_dd", "value"),
    Output("suburb_dd", "options"),
    Output("suburb_dd", "value"),
    Input("region_dd", "value"),
)
def update_type_suburb(region: str):
    region = str(region) if region else DEFAULT_REGION

    dff = DF.loc[DF["Regionname"] == region].copy()
    if dff.empty:
        dff = DF.copy()

    types = sorted(dff["Type"].dropna().astype(str).unique().tolist())
    type_options = [{"label": format_type_option(t), "value": t} for t in types]
    type_val = types[0] if types else None

    suburbs = sorted(dff["Suburb"].dropna().astype(str).unique().tolist())
    suburb_options = [{"label": s, "value": s} for s in suburbs]
    suburb_val = suburbs[0] if suburbs else None

    return type_options, type_val, suburb_options, suburb_val




@app.callback(
    Output("distance_in", "min"),
    Output("distance_in", "max"),
    Output("distance_in", "value"),
    Input("distance_range_dd", "value"),
    State("distance_in", "value"),
)
def constrain_distance_input(distance_range: str, current_distance: float):
    rng = parse_distance_range(distance_range)
    if not rng:
        # fallback: allow a broad range
        return 0.0, float(DF["Distance"].max()), safe_float(current_distance, 12.0)

    lo, hi = rng[0], rng[1]
    # pd.cut used right=False so the bin is [lo, hi)
    # make max slightly below hi to avoid accidental out-of-bin values
    max_val = max(lo, hi - 1e-6)

    # rule: when DistanceRange changes, set Distance to the min of the range
    # user can then adjust within bounds
    return float(lo), float(max_val), float(lo)


@app.callback(
    Output("scatter_graph", "figure"),
    Output("region_box_graph", "figure"),
    Output("results_box", "children"),
    Input("region_dd", "value"),
    Input("distance_range_dd", "value"),
    Input("type_dd", "value"),
    Input("suburb_dd", "value"),
    Input("rooms_in", "value"),
    Input("bath_in", "value"),
    Input("car_in", "value"),
    Input("distance_in", "value"),
    Input("landsize_in", "value"),
)
def update_dashboard(region, distance_range, prop_type, suburb, rooms, bath, car, distance, landsize):
    region = str(region) if region else DEFAULT_REGION
    distance_range = str(distance_range) if distance_range else DEFAULT_DISTANCE_RANGE

    rooms_i = safe_int(rooms, 3)
    bath_i = safe_int(bath, 2)
    car_i = safe_int(car, 1)
    distance_f = safe_float(distance, 12.0)
    landsize_f = safe_float(landsize, 500.0)

    # If dropdowns missing, fallback to first available
    dff_region = DF.loc[DF["Regionname"] == region].copy()
    if dff_region.empty:
        dff_region = DF.copy()

    if prop_type is None and not dff_region["Type"].dropna().empty:
        prop_type = str(dff_region["Type"].dropna().iloc[0])
    if suburb is None and not dff_region["Suburb"].dropna().empty:
        suburb = str(dff_region["Suburb"].dropna().iloc[0])

    # Prediction (log -> exp)
    x_in = pd.DataFrame(
        [
            {
                "Rooms": rooms_i,
                "Bathroom": bath_i,
                "Car": car_i,
                "Distance": distance_f,
                "Landsize": landsize_f,
                "Type": str(prop_type),
                "Suburb": str(suburb),
            }
        ]
    )
    pred_log = float(PIPE.predict(x_in)[0])
    pred_price = float(np.exp(pred_log))

    # Scatter:
    # - Use ALL DistanceRanges in the region so legend has all categories
    # - But set non-selected ranges to legendonly so it looks "filtered" initially
    fig_scatter = px.scatter(
        dff_region,
        x="Landsize",
        y="Price",
        color=dff_region["DistanceRange"].astype(str),
        opacity=0.35,
        title="Exploratory: Price vs Landsize (click legend items to toggle DistanceRanges)",
        hover_data=["Regionname", "Distance", "DistanceRange", "Rooms", "Bathroom", "Car", "Type", "Suburb"],
    )

    # Make only the selected DistanceRange visible by default
    selected_label = str(distance_range)
    for trace in fig_scatter.data:
        if trace.name != selected_label:
            trace.visible = "legendonly"

    # Trendline-like guide for the selected range only (manual polyfit)
    dff_sel = dff_region.loc[dff_region["DistanceRange"].astype(str) == selected_label].copy()
    if len(dff_sel) >= 2:
        xvals = dff_sel["Landsize"].to_numpy(dtype=float)
        yvals = dff_sel["Price"].to_numpy(dtype=float)
        ok = np.isfinite(xvals) & np.isfinite(yvals)
        xvals = xvals[ok]
        yvals = yvals[ok]
        if len(xvals) >= 2:
            m, b0 = np.polyfit(xvals, yvals, 1)
            xs = np.linspace(float(xvals.min()), float(xvals.max()), 60)
            ys = m * xs + b0
            fig_scatter.add_scatter(x=xs, y=ys, mode="lines", name="Trendline", line=dict(width=3))

    # Estimation point always on top
    fig_scatter.add_scatter(
        x=[landsize_f],
        y=[pred_price],
        mode="markers",
        marker=dict(size=18, symbol="x", line=dict(width=2)),
        name="Estimation",
        hovertemplate="Landsize=%{x:.0f} m²<br>Estimated Price=%{y:,.0f}<extra></extra>",
    )

    fig_scatter.update_layout(margin=dict(l=10, r=10, t=60, b=10))

    # Boxplot: reduce outliers per region using IQR; show only selected DistanceRange
    dff_box = DF.loc[DF["DistanceRange"].astype(str) == selected_label].copy()
    if dff_box.empty:
        dff_box = DF.copy()

    cleaned_parts = []
    for reg_name, g in dff_box.groupby("Regionname"):
        cleaned_parts.append(remove_outliers_iqr(g, "Price", k=IQR_K))
    dff_box_clean = pd.concat(cleaned_parts, ignore_index=True) if cleaned_parts else dff_box

    fig_box = px.box(
        dff_box_clean,
        x="Regionname",
        y="Price",
        points=False,
        title="Price distribution by Region (IQR-reduced outliers)",
    )
    fig_box.update_xaxes(tickangle=30)
    fig_box.update_layout(margin=dict(l=10, r=10, t=60, b=10))

    results = [
        html.H4(f"Estimated Price: {format_money_aud(pred_price)}"),
        html.Div(f"Test R² (log-space): {R2_LOG_TEST:.3f}"),
        html.Div(f"Test MAE (approx, AUD): {format_money_aud(MAE_AUD_TEST)}"),
        html.Div("Model: Ridge Regression + one-hot(Type, Suburb) + numeric features"),
        html.Div(f"Filters: Region={region} | DistanceRange={selected_label}"),
    ]

    return fig_scatter, fig_box, results


# ============================================================
# AUTO-OPEN BROWSER
# ============================================================
def open_browser(url: str) -> None:
    try:
        webbrowser.open_new(url)
    except Exception:
        # If it fails, user can still open manually
        pass


if __name__ == "__main__":
    app.run(debug=True)
