import os
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Global Weather Dashboard", page_icon="🌍", layout="wide")

DEFAULT_CSV_PATH = Path("data/GlobalWeatherRepository-processed.csv")
PM25_TARGET_CANDIDATES = ["air_quality_PM2.5", "air_quality_PM2_5", "PM2.5", "pm2_5", "pm25"]
COLUMN_ALIASES = {
    "temperature_c": ["temperature_c", "temperature_celsius"],
    "temperature_f": ["temperature_f", "temperature_fahrenheit"],
    "feelslike_c": ["feelslike_c", "feels_like_celsius"],
    "feelslike_f": ["feelslike_f", "feels_like_fahrenheit"],
    "wind_dir": ["wind_dir", "wind_direction"],
    "uv": ["uv", "uv_index"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for canonical, candidates in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                rename_map[candidate] = canonical
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


@st.cache_data(show_spinner=False)
def load_data(source: Any) -> pd.DataFrame:
    if isinstance(source, str) and source.startswith("s3://"):
        parsed = urlparse(source)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        if not bucket or not key:
            raise ValueError("Invalid s3 URI. Use format: s3://bucket/key.csv")
        try:
            import boto3
        except Exception as exc:
            raise RuntimeError("boto3 is required for s3:// sources. Install dependencies again.") from exc
        s3 = boto3.client("s3")
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        df = pd.read_csv(BytesIO(body))
    else:
        df = pd.read_csv(source)

    df = normalize_columns(df)
    if "last_updated" in df.columns:
        df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
    return df


def sidebar_data_source() -> pd.DataFrame | None:
    st.sidebar.header("Data Source")
    st.sidebar.caption("Use local CSV, AWS endpoint URL, or upload a copy.")

    source = st.sidebar.radio(
        "Choose input",
        options=["Use local CSV", "Use AWS endpoint URL", "Upload CSV"],
        index=0,
    )

    if source == "Use local CSV":
        local_mode = st.sidebar.radio("Local source", ["Browse file", "Path"], index=0)
        if local_mode == "Browse file":
            local_file = st.sidebar.file_uploader("Choose local CSV", type=["csv"], key="local_csv_file")
            if local_file is None:
                return None
            return load_data(local_file)

        path = st.sidebar.text_input("CSV path", value=str(DEFAULT_CSV_PATH))
        if not path:
            return None
        if not os.path.exists(path):
            st.warning(
                f"CSV not found at `{path}`. Download the dataset from Kaggle and place it there, or upload the file."
            )
            return None
        return load_data(path)

    if source == "Use AWS endpoint URL":
        if "aws_csv_url" not in st.session_state:
            st.session_state["aws_csv_url"] = os.getenv("WEATHER_CSV_URL", "")
        endpoint_url = st.sidebar.text_input(
            "CSV URL or S3 URI",
            key="aws_csv_url",
            placeholder="https://your-bucket.s3.amazonaws.com/GlobalWeatherRepository.csv",
            help="Supports HTTP(S) CSV links and s3://bucket/key.csv URIs.",
        )
        if not endpoint_url:
            return None
        if not endpoint_url.lower().startswith(("http://", "https://", "s3://")):
            st.error("Use a direct CSV URL (http/https) or s3://bucket/key.csv, not an endpoint name.")
            return None
        try:
            return load_data(endpoint_url)
        except Exception as exc:
            st.error(f"Could not load CSV from source: {exc}")
            return None

    uploaded_file = st.sidebar.file_uploader("Upload Kaggle CSV", type=["csv"])
    if uploaded_file is None:
        return None
    return load_data(uploaded_file)


def build_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Sampling")
    sample_mode = st.sidebar.radio("Dataset mode", ["Full dataset", "Random sample"], index=0)
    if sample_mode == "Random sample":
        max_rows = int(len(df))
        min_rows = 1 if max_rows < 100 else 100
        default_rows = min(5000, max_rows)
        step = 1 if max_rows < 100 else 100
        sample_rows = st.sidebar.slider("Sample rows", min_value=min_rows, max_value=max_rows, value=default_rows, step=step)
        random_seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
        if sample_rows < len(df):
            df = df.sample(n=sample_rows, random_state=int(random_seed))
        st.sidebar.caption(f"Using random sample: {len(df):,} rows")
    else:
        st.sidebar.caption(f"Using full dataset: {len(df):,} rows")

    st.sidebar.header("Filters")

    filtered = df.copy()

    if "country" in filtered.columns:
        countries = sorted(filtered["country"].dropna().unique().tolist())
        selected_countries = st.sidebar.multiselect("Country (optional)", options=countries, default=[])
        if selected_countries:
            filtered = filtered[filtered["country"].isin(selected_countries)]

    if "condition_text" in filtered.columns:
        conditions = sorted(filtered["condition_text"].dropna().unique().tolist())
        selected_conditions = st.sidebar.multiselect("Weather condition", options=conditions)
        if selected_conditions:
            filtered = filtered[filtered["condition_text"].isin(selected_conditions)]

    if "temperature_c" in filtered.columns:
        min_temp = float(filtered["temperature_c"].min())
        max_temp = float(filtered["temperature_c"].max())
        temp_range = st.sidebar.slider("Temperature (C)", min_value=min_temp, max_value=max_temp, value=(min_temp, max_temp))
        filtered = filtered[(filtered["temperature_c"] >= temp_range[0]) & (filtered["temperature_c"] <= temp_range[1])]

    if "last_updated" in filtered.columns and pd.api.types.is_datetime64_any_dtype(filtered["last_updated"]):
        valid_dates = filtered["last_updated"].dropna()
        if not valid_dates.empty:
            start_default = valid_dates.min().date()
            end_default = valid_dates.max().date()
            selected_dates = st.sidebar.date_input(
                "Date range",
                value=(start_default, end_default),
                min_value=start_default,
                max_value=end_default,
            )
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
                filtered = filtered[
                    (filtered["last_updated"].dt.date >= start_date)
                    & (filtered["last_updated"].dt.date <= end_date)
                ]

    with st.sidebar.expander("Advanced filters"):
        extra_categorical = [
            "location_name",
            "region",
            "timezone",
            "wind_dir",
        ]
        for col in extra_categorical:
            if col in filtered.columns:
                options = sorted(filtered[col].dropna().astype(str).unique().tolist())
                if options:
                    selected = st.multiselect(col.replace("_", " ").title(), options=options)
                    if selected:
                        filtered = filtered[filtered[col].astype(str).isin(selected)]

        numeric_candidates = [
            "feelslike_c",
            "wind_kph",
            "gust_kph",
            "pressure_mb",
            "precip_mm",
            "visibility_km",
            "uv",
            "cloud",
            "humidity",
            "air_quality_Carbon_Monoxide",
            "air_quality_Ozone",
            "air_quality_Nitrogen_dioxide",
            "air_quality_Sulphur_dioxide",
            "air_quality_PM2.5",
            "air_quality_PM10",
        ]

        for col in numeric_candidates:
            if col in filtered.columns and pd.api.types.is_numeric_dtype(filtered[col]):
                valid = filtered[col].dropna()
                if valid.empty:
                    continue
                min_val = float(valid.min())
                max_val = float(valid.max())
                if min_val == max_val:
                    continue
                step = max((max_val - min_val) / 100.0, 0.01)
                label = f"{col.replace('_', ' ')}"
                selected_range = st.slider(
                    label,
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=step,
                )
                filtered = filtered[(filtered[col] >= selected_range[0]) & (filtered[col] <= selected_range[1])]

    return filtered


def metric_cards(df: pd.DataFrame) -> None:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Countries", f"{df['country'].nunique() if 'country' in df.columns else 0:,}")
    col3.metric(
        "Avg Temp (C)",
        f"{df['temperature_c'].mean():.1f}" if "temperature_c" in df.columns and not df.empty else "N/A",
    )
    col4.metric(
        "Avg Humidity",
        f"{df['humidity'].mean():.0f}%" if "humidity" in df.columns and not df.empty else "N/A",
    )


def chart_section(df: pd.DataFrame) -> None:
    left, right = st.columns((1.2, 1))

    with left:
        st.subheader("Temperature by Country")
        if {"country", "temperature_c"}.issubset(df.columns):
            by_country = (
                df.groupby("country", as_index=False)["temperature_c"]
                .mean()
                .sort_values("temperature_c", ascending=False)
                .head(20)
            )
            fig_country = px.bar(
                by_country,
                x="country",
                y="temperature_c",
                color="temperature_c",
                color_continuous_scale="RdYlBu_r",
                title="Top 20 warmest countries (average)",
            )
            fig_country.update_layout(xaxis_title="Country", yaxis_title="Temperature (C)")
            st.plotly_chart(fig_country, use_container_width=True)
        else:
            st.info("Required columns not found for country temperature chart.")

    with right:
        st.subheader("Condition Mix")
        if "condition_text" in df.columns:
            condition_counts = df["condition_text"].value_counts().head(10).reset_index()
            condition_counts.columns = ["condition_text", "count"]
            fig_pie = px.pie(condition_counts, values="count", names="condition_text", hole=0.35)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Required column not found for condition chart.")


def pm25_visual_section(df: pd.DataFrame) -> None:
    st.subheader("PM2.5 Insights")
    target_col = find_pm25_target(df)
    if target_col is None:
        st.info("PM2.5 column not found for visualization.")
        return

    pm_df = df.dropna(subset=[target_col]).copy()
    if pm_df.empty:
        st.info("No non-null PM2.5 values available for visualization.")
        return

    severity_bins = [-float("inf"), 12, 35.4, 55.4, 150.4, 250.4, float("inf")]
    severity_labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    pm_df["pm25_severity"] = pd.cut(pm_df[target_col], bins=severity_bins, labels=severity_labels)

    top_left, top_right = st.columns(2)
    with top_left:
        fig_hist = px.histogram(
            pm_df,
            x=target_col,
            nbins=50,
            title="PM2.5 Distribution",
            labels={target_col: "PM2.5"},
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with top_right:
        if "country" in pm_df.columns:
            by_country_pm = (
                pm_df.groupby("country", as_index=False)[target_col]
                .mean()
                .sort_values(target_col, ascending=False)
                .head(15)
            )
            fig_country_pm = px.bar(
                by_country_pm,
                x="country",
                y=target_col,
                color=target_col,
                color_continuous_scale="OrRd",
                title="Top 15 Countries by Average PM2.5",
                labels={target_col: "Avg PM2.5"},
            )
            st.plotly_chart(fig_country_pm, use_container_width=True)
        else:
            st.info("Country column not found for PM2.5 country ranking.")

    mid_left, mid_right = st.columns(2)
    with mid_left:
        severity_counts = (
            pm_df["pm25_severity"]
            .value_counts(dropna=False)
            .rename_axis("severity")
            .reset_index(name="count")
        )
        if not severity_counts.empty:
            fig_severity = px.pie(
                severity_counts,
                names="severity",
                values="count",
                hole=0.45,
                title="PM2.5 Severity Distribution",
            )
            st.plotly_chart(fig_severity, use_container_width=True)
        else:
            st.info("PM2.5 severity distribution unavailable.")

    with mid_right:
        scatter_feature = None
        for candidate in ["humidity", "temperature_c", "wind_kph", "pressure_mb"]:
            if candidate in pm_df.columns and pd.api.types.is_numeric_dtype(pm_df[candidate]):
                scatter_feature = candidate
                break
        if scatter_feature is not None:
            fig_scatter = px.scatter(
                pm_df,
                x=scatter_feature,
                y=target_col,
                color="pm25_severity",
                title=f"PM2.5 vs {scatter_feature.replace('_', ' ').title()}",
                labels={scatter_feature: scatter_feature.replace("_", " ").title(), target_col: "PM2.5"},
                opacity=0.45,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("No numeric weather feature available for PM2.5 comparison scatter.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        if "last_updated" in pm_df.columns and pd.api.types.is_datetime64_any_dtype(pm_df["last_updated"]):
            trend = (
                pm_df.dropna(subset=["last_updated"])
                .assign(date_only=pm_df["last_updated"].dt.date)
                .groupby("date_only", as_index=False)[target_col]
                .mean()
                .sort_values("date_only")
            )
            if not trend.empty:
                fig_trend = px.line(
                    trend,
                    x="date_only",
                    y=target_col,
                    title="Daily Average PM2.5 Trend",
                    labels={"date_only": "Date", target_col: "Avg PM2.5"},
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No valid timestamps available for PM2.5 trend.")
        else:
            st.info("`last_updated` not available for PM2.5 trend chart.")

    with bottom_right:
        if {"latitude", "longitude"}.issubset(pm_df.columns):
            geo_df = pm_df.dropna(subset=["latitude", "longitude"]).copy()
            if not geo_df.empty:
                geo_sample = geo_df.sample(min(len(geo_df), 5000), random_state=42)
                fig_geo = px.scatter_mapbox(
                    geo_sample,
                    lat="latitude",
                    lon="longitude",
                    color=target_col,
                    size=target_col,
                    hover_name="location_name" if "location_name" in geo_sample.columns else None,
                    hover_data={"country": True} if "country" in geo_sample.columns else None,
                    title="PM2.5 Geographic Distribution",
                    color_continuous_scale="YlOrRd",
                    zoom=1,
                    mapbox_style="carto-positron",
                )
                st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.info("No latitude/longitude rows available for PM2.5 geo chart.")
        else:
            st.info("Latitude/longitude columns not available for PM2.5 geo chart.")

    extra_left, extra_right = st.columns(2)
    with extra_left:
        numeric_cols = [c for c in pm_df.select_dtypes(include=["number"]).columns if c != target_col]
        if numeric_cols:
            corr = (
                pm_df[numeric_cols + [target_col]]
                .corr(numeric_only=True)[target_col]
                .drop(target_col)
                .dropna()
                .abs()
                .sort_values(ascending=False)
                .head(10)
                .rename_axis("feature")
                .reset_index(name="abs_corr")
            )
            if not corr.empty:
                fig_corr = px.bar(
                    corr,
                    x="feature",
                    y="abs_corr",
                    title="Top Features Correlated with PM2.5",
                    labels={"feature": "Feature", "abs_corr": "Absolute Correlation"},
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Correlation chart unavailable.")
        else:
            st.info("No numeric features available for correlation chart.")

    with extra_right:
        if (
            "last_updated" in pm_df.columns
            and pd.api.types.is_datetime64_any_dtype(pm_df["last_updated"])
            and "country" in pm_df.columns
        ):
            heat_df = pm_df.dropna(subset=["last_updated", "country"]).copy()
            if not heat_df.empty:
                heat_df["month"] = heat_df["last_updated"].dt.to_period("M").astype(str)
                top_countries = (
                    heat_df.groupby("country")[target_col]
                    .mean()
                    .sort_values(ascending=False)
                    .head(12)
                    .index
                    .tolist()
                )
                heat_df = heat_df[heat_df["country"].isin(top_countries)]
                pivot = heat_df.pivot_table(
                    index="country",
                    columns="month",
                    values=target_col,
                    aggfunc="mean",
                )
                if not pivot.empty:
                    fig_heat = px.imshow(
                        pivot,
                        aspect="auto",
                        color_continuous_scale="YlOrRd",
                        title="Country vs Month PM2.5 Heatmap",
                        labels={"x": "Month", "y": "Country", "color": "Avg PM2.5"},
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)
                else:
                    st.info("Heatmap unavailable after filtering.")
            else:
                st.info("No rows available for PM2.5 heatmap.")
        else:
            st.info("Need `country` and `last_updated` for PM2.5 heatmap.")


def table_section(df: pd.DataFrame) -> None:
    st.subheader("Raw Data")

    preferred_columns = [
        "country",
        "location_name",
        "last_updated",
        "temperature_c",
        "humidity",
        "condition_text",
        "wind_kph",
        "precip_mm",
    ]
    existing_columns = [c for c in preferred_columns if c in df.columns]

    st.dataframe(df[existing_columns] if existing_columns else df, use_container_width=True, hide_index=True)


def find_pm25_target(df: pd.DataFrame) -> str | None:
    for candidate in PM25_TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def train_pm25_model(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    model_df = df.dropna(subset=[target_col]).copy()

    # Remove columns that are not useful for tabular regression.
    excluded = {target_col, "last_updated", "sunrise", "sunset", "moonrise", "moonset"}
    feature_df = model_df[[c for c in model_df.columns if c not in excluded]].copy()

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if feature_df[c].nunique(dropna=True) <= 80]

    selected_cols = numeric_cols + categorical_cols
    if not selected_cols:
        raise ValueError("No usable features available for training.")

    X = feature_df[selected_cols]
    y = model_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model_defs: list[tuple[str, Any]] = [
        ("LinearRegression", LinearRegression()),
        ("RandomForest", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
        ("GradientBoosting", GradientBoostingRegressor(random_state=42)),
    ]

    rows: list[dict[str, Any]] = []
    best_r2 = float("-inf")
    best_name = ""
    best_model = None
    best_predictions = None

    for model_name, regressor in model_defs:
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor),
            ]
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        rows.append({"model": model_name, "mae": mae, "r2": r2})

        if r2 > best_r2:
            best_r2 = r2
            best_name = model_name
            best_model = model
            best_predictions = predictions

    return {
        "model": best_model,
        "best_model_name": best_name,
        "X": X,
        "defaults": {
            col: (float(X[col].median()) if pd.api.types.is_numeric_dtype(X[col]) else (X[col].dropna().astype(str).mode().iloc[0] if not X[col].dropna().empty else ""))
            for col in X.columns
        },
        "scores": pd.DataFrame(rows).sort_values("r2", ascending=False),
        "target_col": target_col,
        "actual_vs_pred": pd.DataFrame({"actual": y_test, "predicted": best_predictions}),
    }


def pm25_prediction_tab(df: pd.DataFrame) -> None:
    st.subheader("PM2.5 Prediction")

    target_col = find_pm25_target(df)
    if target_col is None:
        st.warning(
            "PM2.5 column not found. Expected one of: "
            + ", ".join(PM25_TARGET_CANDIDATES)
        )
        return

    if df[target_col].dropna().shape[0] < 100:
        st.warning("Not enough PM2.5 rows to train a reliable model.")
        return

    with st.spinner("Training PM2.5 model..."):
        trained = train_pm25_model(df, target_col)

    metric_col1, metric_col2 = st.columns(2)
    best_row = trained["scores"].iloc[0]
    metric_col1.metric("Best Model", str(trained["best_model_name"]))
    metric_col2.metric("Best R²", f"{best_row['r2']:.3f}")

    st.caption(
        "Models tested: Linear Regression, Random Forest, Gradient Boosting. "
        f"Best model selected: {trained['best_model_name']}. "
        "Use this for exploration, not regulatory forecasting."
    )
    st.dataframe(
        trained["scores"].rename(columns={"mae": "MAE", "r2": "R2"}).reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )

    comparison_fig = px.scatter(
        trained["actual_vs_pred"],
        x="actual",
        y="predicted",
        title="Actual vs Predicted PM2.5 (test split)",
        labels={"actual": "Actual PM2.5", "predicted": "Predicted PM2.5"},
        opacity=0.5,
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

    st.markdown("### Predict a single sample")
    X = trained["X"]

    preferred_inputs = [
        "temperature_c",
        "humidity",
        "wind_kph",
        "pressure_mb",
        "precip_mm",
        "uv_index",
        "uv",
        "cloud",
        "visibility_km",
        "condition_text",
        "country",
    ]
    input_columns = [c for c in preferred_inputs if c in X.columns]
    if not input_columns:
        input_columns = X.columns.tolist()[:8]

    with st.form("pm25_predict_form"):
        sample: dict[str, Any] = {}
        for col in input_columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                default_value = float(X[col].median())
                sample[col] = st.number_input(col, value=default_value)
            else:
                options = X[col].dropna().astype(str).value_counts().head(25).index.tolist()
                if not options:
                    options = [""]
                sample[col] = st.selectbox(col, options=options)

        submitted = st.form_submit_button("Predict PM2.5")

    if submitted:
        input_row = pd.DataFrame([sample])
        for col in X.columns:
            if col not in input_row.columns:
                input_row[col] = trained["defaults"].get(col, np.nan)
        input_row = input_row[X.columns]
        try:
            prediction = trained["model"].predict(input_row)[0]
            st.success(f"Predicted PM2.5: {prediction:.2f}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def main() -> None:
    st.title("Global Weather Dashboard")
    st.caption("Built with Streamlit using the Kaggle Global Weather Repository dataset")

    st.markdown(
        "Dataset: [Global Weather Repository on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/data)"
    )

    df = sidebar_data_source()

    if df is None:
        st.info(
            "No dataset loaded yet. Put the Kaggle CSV at `data/GlobalWeatherRepository.csv` or upload it from the sidebar."
        )
        return

    filtered_df = build_filters(df)

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    tab_dashboard, tab_predict = st.tabs(["Dashboard", "PM2.5 Predictor"])

    with tab_dashboard:
        metric_cards(filtered_df)
        chart_section(filtered_df)
        pm25_visual_section(filtered_df)
        table_section(filtered_df)

    with tab_predict:
        pm25_prediction_tab(filtered_df)


if __name__ == "__main__":
    main()
