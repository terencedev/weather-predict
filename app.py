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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Global Weather Dashboard", page_icon="🌍", layout="wide")

DEFAULT_CSV_PATH = Path("data/GlobalWeatherRepository-processed.csv")
PM25_TARGET_CANDIDATES = ["air_quality_PM2.5", "air_quality_PM2_5", "PM2.5", "pm2_5", "pm25"]
AQI_TARGET_CANDIDATES = ["air_quality_us_epa_index", "aqi", "AQI"]
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
    st.sidebar.caption("Using default local CSV source.")

    path = str(DEFAULT_CSV_PATH)
    st.sidebar.text_input("CSV path", value=path, disabled=True)

    if not os.path.exists(path):
        st.warning(f"CSV not found at `{path}`.")
        return None
    return load_data(path)


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
                fig_geo = px.scatter_map(
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
                    map_style="carto-positron",
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


def find_aqi_target(df: pd.DataFrame) -> str | None:
    for candidate in AQI_TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def reference_dashboard_section(df: pd.DataFrame) -> None:
    st.subheader("QuickSight")

    pm25_col = find_pm25_target(df)
    if pm25_col is None:
        st.info("PM2.5 column not found. Reference dashboard needs PM2.5 data.")
        return

    ref_df = df.dropna(subset=[pm25_col]).copy()
    if ref_df.empty:
        st.info("No PM2.5 rows available for reference dashboard.")
        return

    aqi_col = find_aqi_target(ref_df)

    severity_bins = [-float("inf"), 12, 35.4, 55.4, 150.4, 250.4, float("inf")]
    severity_labels = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]
    ref_df["pm25_severity"] = pd.cut(ref_df[pm25_col], bins=severity_bins, labels=severity_labels)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average PM2.5 Concentration", f"{ref_df[pm25_col].mean():.2f}")
    if aqi_col is not None:
        col2.metric("Average AQI (US EPA)", f"{pd.to_numeric(ref_df[aqi_col], errors='coerce').mean():.2f}")
    else:
        col2.metric("Average AQI (US EPA)", "N/A")
    col3.metric(
        "% Unhealthy / Hazardous Days",
        f"{((ref_df[pm25_col] > 55.4).mean() * 100):.2f}",
    )
    if "location_name" in ref_df.columns:
        top_city = (
            ref_df.groupby("location_name", as_index=False)[pm25_col]
            .mean()
            .sort_values(pm25_col, ascending=False)
            .head(1)
        )
        if not top_city.empty:
            col4.metric("Most Polluted City", f"{top_city.iloc[0]['location_name']} ({top_city.iloc[0][pm25_col]:.1f})")
        else:
            col4.metric("Most Polluted City", "N/A")
    else:
        col4.metric("Most Polluted City", "N/A")

    row1_left, row1_right = st.columns(2)
    with row1_left:
        if {"latitude", "longitude"}.issubset(ref_df.columns):
            geo_df = ref_df.dropna(subset=["latitude", "longitude"]).copy()
            if not geo_df.empty:
                geo_sample = geo_df.sample(min(len(geo_df), 2500), random_state=42)
                fig_geo = px.scatter_map(
                    geo_sample,
                    lat="latitude",
                    lon="longitude",
                    color="pm25_severity",
                    size=pm25_col,
                    hover_name="location_name" if "location_name" in geo_sample.columns else None,
                    hover_data={"country": True} if "country" in geo_sample.columns else None,
                    title="Air Quality Across the World",
                    map_style="carto-positron",
                    zoom=1,
                )
                st.plotly_chart(fig_geo, use_container_width=True)
            else:
                st.info("No latitude/longitude rows available.")
        else:
            st.info("Latitude/longitude columns not available.")

    with row1_right:
        if "air_quality_PM10" in ref_df.columns:
            scatter_df = ref_df.dropna(subset=["air_quality_PM10", pm25_col]).copy()
            if not scatter_df.empty:
                scatter_df = scatter_df.sample(min(len(scatter_df), 2500), random_state=42)
                fig_pm10 = px.scatter(
                    scatter_df,
                    x="air_quality_PM10",
                    y=pm25_col,
                    size=pm25_col,
                    color="pm25_severity",
                    hover_name="location_name" if "location_name" in scatter_df.columns else None,
                    title="PM10 vs PM2.5",
                    labels={"air_quality_PM10": "PM10", pm25_col: "PM2.5"},
                    opacity=0.6,
                )
                st.plotly_chart(fig_pm10, use_container_width=True)
            else:
                st.info("No PM10/PM2.5 rows available.")
        else:
            st.info("PM10 column not found.")

    row2_left, row2_right = st.columns(2)
    with row2_left:
        if "wind_kph" in ref_df.columns:
            wind_df = ref_df.dropna(subset=["wind_kph", pm25_col]).copy()
            if not wind_df.empty:
                wind_sample = wind_df.sample(min(len(wind_df), 2500), random_state=42)
                fig_wind = px.scatter(
                    wind_sample,
                    x=pm25_col,
                    y="wind_kph",
                    color="country" if "country" in wind_sample.columns else None,
                    size=pm25_col,
                    title="Wind Speed vs PM2.5",
                    labels={pm25_col: "PM2.5", "wind_kph": "Wind (kph)"},
                    opacity=0.65,
                )
                st.plotly_chart(fig_wind, use_container_width=True)
            else:
                st.info("No wind/PM2.5 rows available.")
        else:
            st.info("Wind column not found.")

    with row2_right:
        if "wind_kph" in ref_df.columns:
            wind_cat_df = ref_df.dropna(subset=["wind_kph", pm25_col]).copy()
            if not wind_cat_df.empty:
                wind_cat_df["wind_category"] = pd.cut(
                    wind_cat_df["wind_kph"],
                    bins=[-float("inf"), 10, 25, float("inf")],
                    labels=["Low Wind", "Medium Wind", "High Wind"],
                )
                by_wind = (
                    wind_cat_df.groupby("wind_category", as_index=False)[pm25_col]
                    .mean()
                    .sort_values("wind_category")
                )
                fig_wind_bar = px.bar(
                    by_wind,
                    x="wind_category",
                    y=pm25_col,
                    title="Average PM2.5 by Wind Category",
                    labels={"wind_category": "Wind Category", pm25_col: "PM2.5 (Average)"},
                )
                st.plotly_chart(fig_wind_bar, use_container_width=True)
            else:
                st.info("No wind/PM2.5 rows available.")
        else:
            st.info("Wind column not found.")

    row3_left, row3_right = st.columns(2)
    with row3_left:
        if "humidity" in ref_df.columns:
            hum_df = ref_df.dropna(subset=["humidity", pm25_col]).copy()
            if not hum_df.empty:
                hum_df["humidity_category"] = pd.cut(
                    hum_df["humidity"],
                    bins=[-float("inf"), 40, 70, float("inf")],
                    labels=["Low (<40%)", "Medium (40-70%)", "High (>70%)"],
                )
                by_hum = hum_df.groupby("humidity_category", as_index=False)[pm25_col].mean()
                fig_hum = px.bar(
                    by_hum,
                    x="humidity_category",
                    y=pm25_col,
                    title="Average PM2.5 by Humidity Category",
                    labels={"humidity_category": "Humidity Category", pm25_col: "PM2.5 (Average)"},
                )
                st.plotly_chart(fig_hum, use_container_width=True)
            else:
                st.info("No humidity/PM2.5 rows available.")
        else:
            st.info("Humidity column not found.")

    with row3_right:
        if "temperature_c" in ref_df.columns:
            temp_df = ref_df.dropna(subset=["temperature_c", pm25_col]).copy()
            if not temp_df.empty:
                temp_df = temp_df.sample(min(len(temp_df), 2500), random_state=42)
                fig_temp = px.scatter(
                    temp_df,
                    x="temperature_c",
                    y=pm25_col,
                    title="Temperature vs PM2.5",
                    labels={"temperature_c": "Temperature (C)", pm25_col: "PM2.5"},
                    opacity=0.55,
                )
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.info("No temperature/PM2.5 rows available.")
        else:
            st.info("Temperature column not found.")

    row4_left, row4_right = st.columns(2)
    with row4_left:
        if "last_updated" in ref_df.columns and pd.api.types.is_datetime64_any_dtype(ref_df["last_updated"]):
            trend_df = (
                ref_df.dropna(subset=["last_updated"])
                .assign(month=ref_df["last_updated"].dt.to_period("M").astype(str))
                .groupby("month", as_index=False)[pm25_col]
                .mean()
            )
            if not trend_df.empty:
                fig_trend = px.line(
                    trend_df,
                    x="month",
                    y=pm25_col,
                    title="Average PM2.5 by Month",
                    labels={"month": "Month", pm25_col: "PM2.5 (Average)"},
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No valid timestamps for PM2.5 trend.")
        else:
            st.info("`last_updated` column not available for trend.")

    with row4_right:
        if "location_name" in ref_df.columns:
            by_city = (
                ref_df.groupby("location_name", as_index=False)[pm25_col]
                .mean()
                .sort_values(pm25_col, ascending=False)
                .head(25)
            )
            if not by_city.empty:
                fig_city = px.bar(
                    by_city.sort_values(pm25_col, ascending=True),
                    x=pm25_col,
                    y="location_name",
                    orientation="h",
                    title="Most Polluted City (Top 25 by Avg PM2.5)",
                    labels={"location_name": "City", pm25_col: "PM2.5 (Average)"},
                )
                st.plotly_chart(fig_city, use_container_width=True)
            else:
                st.info("No city rows available for PM2.5 ranking.")
        else:
            st.info("`location_name` column not found.")

    if aqi_col is not None:
        aqi_df = ref_df.dropna(subset=[aqi_col]).copy()
        if not aqi_df.empty:
            aqi_vals = pd.to_numeric(aqi_df[aqi_col], errors="coerce").clip(lower=1, upper=6).round()
            aqi_map = {
                1: "Good",
                2: "Moderate",
                3: "Unhealthy for Sensitive Groups",
                4: "Unhealthy",
                5: "Very Unhealthy",
                6: "Hazardous",
            }
            aqi_df["aqi_label"] = aqi_vals.map(aqi_map)

            row5_left, row5_right = st.columns(2)
            with row5_left:
                aqi_counts = (
                    aqi_df["aqi_label"]
                    .value_counts()
                    .rename_axis("aqi_label")
                    .reset_index(name="count")
                )
                if not aqi_counts.empty:
                    fig_aqi_pie = px.pie(
                        aqi_counts,
                        names="aqi_label",
                        values="count",
                        hole=0.45,
                        title="Average AQI Distribution",
                    )
                    st.plotly_chart(fig_aqi_pie, use_container_width=True)
                else:
                    st.info("AQI distribution unavailable.")

            with row5_right:
                if "country" in aqi_df.columns:
                    country_aqi = (
                        aqi_df.groupby(["country", "aqi_label"])
                        .size()
                        .reset_index(name="count")
                    )
                    top_countries = (
                        country_aqi.groupby("country", as_index=False)["count"]
                        .sum()
                        .sort_values("count", ascending=False)
                        .head(12)["country"]
                        .tolist()
                    )
                    country_aqi = country_aqi[country_aqi["country"].isin(top_countries)]
                    fig_aqi_stack = px.bar(
                        country_aqi,
                        x="country",
                        y="count",
                        color="aqi_label",
                        title="Distribution of AQI Levels by Country",
                        labels={"count": "Count", "country": "Country", "aqi_label": "AQI"},
                    )
                    st.plotly_chart(fig_aqi_stack, use_container_width=True)
                else:
                    st.info("Country column not found for AQI country chart.")


@st.cache_resource(show_spinner=False)
def train_epa_index_classifier(df: pd.DataFrame) -> dict[str, Any]:
    aqi_col = find_aqi_target(df)
    if aqi_col is None:
        raise ValueError("No AQI column available. Expected `air_quality_us_epa_index`.")

    model_df = df.copy()
    aqi_vals = pd.to_numeric(model_df[aqi_col], errors="coerce").clip(lower=1, upper=6).round()
    model_df["aqi_class"] = aqi_vals.astype("Int64").astype(str)
    model_df = model_df[model_df["aqi_class"] != "<NA>"].copy()

    if len(model_df) < 300:
        raise ValueError("Not enough labeled rows for EPA classification.")

    excluded = {"aqi_class", aqi_col, "last_updated", "sunrise", "sunset", "moonrise", "moonset"}
    pm25_col = find_pm25_target(model_df)
    if pm25_col is not None:
        excluded.add(pm25_col)

    feature_df = model_df[[c for c in model_df.columns if c not in excluded]].copy()
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if feature_df[c].nunique(dropna=True) <= 100]

    selected_cols = numeric_cols + categorical_cols
    if not selected_cols:
        raise ValueError("No usable features available for EPA classification.")

    X = feature_df[selected_cols]
    y = model_df["aqi_class"]
    if y.nunique() < 2:
        raise ValueError("EPA classification requires at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "actual_vs_pred": pd.DataFrame({"actual": y_test.values, "predicted": predictions}),
        "class_counts": y.value_counts().sort_index().rename_axis("class").reset_index(name="count"),
    }


def epa_index_classifier_tab(df: pd.DataFrame) -> None:
    st.subheader("EPA Index Classifier")
    st.caption("Multiclass classification for `air_quality_us_epa_index` (classes 1-6).")

    try:
        with st.spinner("Training EPA index classifier..."):
            trained = train_epa_index_classifier(df)
    except Exception as exc:
        st.warning(f"EPA classifier unavailable: {exc}")
        return

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{trained['accuracy']:.3f}")
    c2.metric("F1 (macro)", f"{trained['f1_macro']:.3f}")

    st.plotly_chart(
        px.bar(
            trained["class_counts"],
            x="class",
            y="count",
            title="EPA Class Distribution",
            labels={"class": "EPA Class", "count": "Rows"},
        ),
        use_container_width=True,
    )

    compare = trained["actual_vs_pred"].copy()
    compare["pair"] = compare["actual"] + " -> " + compare["predicted"]
    pair_counts = compare["pair"].value_counts().head(20).rename_axis("pair").reset_index(name="count")
    st.plotly_chart(
        px.bar(
            pair_counts.sort_values("count", ascending=True),
            x="count",
            y="pair",
            orientation="h",
            title="Top Outcomes (Actual -> Predicted)",
            labels={"count": "Rows", "pair": "Outcome"},
        ),
        use_container_width=True,
    )


@st.cache_resource(show_spinner=False)
def train_unhealthy_risk_classifier(df: pd.DataFrame, threshold: float) -> dict[str, Any]:
    pm25_col = find_pm25_target(df)
    if pm25_col is None:
        raise ValueError("No PM2.5 column available for risk classification.")

    model_df = df.copy()
    pm25_vals = pd.to_numeric(model_df[pm25_col], errors="coerce")
    model_df = model_df[pm25_vals.notna()].copy()
    model_df["risk_label"] = np.where(pm25_vals[pm25_vals.notna()] > threshold, "Unhealthy", "Healthy")

    if len(model_df) < 300:
        raise ValueError("Not enough rows for risk classification.")
    if model_df["risk_label"].nunique() < 2:
        raise ValueError("Only one class present after thresholding. Change threshold.")

    excluded = {"risk_label", pm25_col, "last_updated", "sunrise", "sunset", "moonrise", "moonset"}
    aqi_col = find_aqi_target(model_df)
    if aqi_col is not None:
        excluded.add(aqi_col)

    feature_df = model_df[[c for c in model_df.columns if c not in excluded]].copy()
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if feature_df[c].nunique(dropna=True) <= 100]

    selected_cols = numeric_cols + categorical_cols
    if not selected_cols:
        raise ValueError("No usable features available for risk classification.")

    X = feature_df[selected_cols]
    y = model_df["risk_label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

    classifier = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)),
        ]
    )
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "f1_macro": f1_score(y_test, predictions, average="macro"),
        "actual_vs_pred": pd.DataFrame({"actual": y_test.values, "predicted": predictions}),
        "class_counts": y.value_counts().rename_axis("class").reset_index(name="count"),
    }


def unhealthy_risk_classifier_tab(df: pd.DataFrame) -> None:
    st.subheader("Unhealthy Risk Classifier")
    st.caption("Binary classification for PM2.5 unhealthy risk.")
    threshold = st.selectbox(
        "Risk threshold (PM2.5)",
        options=[35.4, 55.4],
        index=1,
        help="35.4 = USG threshold, 55.4 = Unhealthy threshold.",
    )

    try:
        with st.spinner("Training unhealthy risk classifier..."):
            trained = train_unhealthy_risk_classifier(df, float(threshold))
    except Exception as exc:
        st.warning(f"Risk classifier unavailable: {exc}")
        return

    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{trained['accuracy']:.3f}")
    c2.metric("F1 (macro)", f"{trained['f1_macro']:.3f}")

    st.plotly_chart(
        px.bar(
            trained["class_counts"],
            x="class",
            y="count",
            title=f"Risk Class Distribution (threshold {threshold})",
            labels={"class": "Class", "count": "Rows"},
        ),
        use_container_width=True,
    )

    compare = trained["actual_vs_pred"].copy()
    compare["pair"] = compare["actual"] + " -> " + compare["predicted"]
    pair_counts = compare["pair"].value_counts().rename_axis("pair").reset_index(name="count")
    st.plotly_chart(
        px.bar(
            pair_counts.sort_values("count", ascending=True),
            x="count",
            y="pair",
            orientation="h",
            title="Prediction Outcomes (Actual -> Predicted)",
            labels={"count": "Rows", "pair": "Outcome"},
        ),
        use_container_width=True,
    )


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

    tab_dashboard, tab_quicksight, tab_predict, tab_epa, tab_risk = st.tabs(
        ["Dashboard", "QuickSight", "PM2.5 Predictor", "EPA Index Classifier", "Unhealthy Risk Classifier"]
    )

    with tab_dashboard:
        metric_cards(filtered_df)
        chart_section(filtered_df)
        pm25_visual_section(filtered_df)
        table_section(filtered_df)

    with tab_quicksight:
        try:
            reference_dashboard_section(filtered_df)
        except Exception as exc:
            st.error(f"QuickSight error: {exc}")

    with tab_predict:
        pm25_prediction_tab(filtered_df)

    with tab_epa:
        epa_index_classifier_tab(filtered_df)

    with tab_risk:
        unhealthy_risk_classifier_tab(filtered_df)


if __name__ == "__main__":
    main()
