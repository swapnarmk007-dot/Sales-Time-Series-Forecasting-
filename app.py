import os
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ModuleNotFoundError:
    JOBLIB_AVAILABLE = False

# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Time Series Forecasting")
st.subheader("Retail Sales Prediction Dashboard")

st.write(
    "Analyze historical sales and forecast future sales "
    "using Machine Learning and ARIMA."
)

# --------------------------------
# 2. LOAD DATASET
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

POSSIBLE_DATA_PATHS = [
    os.path.join(BASE_DIR, "dataset", "retail_store_inventory.csv"),
    os.path.join(BASE_DIR, "retail_store_inventory.csv"),
    os.path.join(os.path.dirname(BASE_DIR), "dataset", "retail_store_inventory.csv"),
]

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "arima_sales_model.pkl"
)

found_data_path = next((p for p in POSSIBLE_DATA_PATHS if os.path.exists(p)), None)

if found_data_path:
    df = pd.read_csv(found_data_path)
    st.caption(f"Loaded dataset from: `{found_data_path}`")
else:
    st.warning("`retail_store_inventory.csv` was not found in the expected dataset folder.")
    st.info(
        "Place the file at:\n"
        f"`{POSSIBLE_DATA_PATHS[0]}`\n\n"
        "Or upload the CSV file below to continue:"
    )
    uploaded_file = st.file_uploader("Upload retail_store_inventory.csv", type=["csv"])
    if uploaded_file is None:
        st.stop()
    df = pd.read_csv(uploaded_file)

# Validate expected columns before preprocessing
required_cols = {"Date", "Units Sold"}
if not required_cols.issubset(df.columns):
    st.error(
        f"Missing required columns: {sorted(required_cols - set(df.columns))}. "
        f"Found columns: {list(df.columns)}"
    )
    st.stop()

# --------------------------------
# 3. DATA PREPROCESSING
# --------------------------------

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", "Units Sold"])

daily_sales = (
    df.groupby("Date")["Units Sold"]
    .sum()
    .sort_index()
)

daily_sales = daily_sales.asfreq("D")
daily_sales = daily_sales.interpolate(method="time")

if daily_sales.empty:
    st.error("No valid sales rows remain after cleaning Date and Units Sold.")
    st.stop()

# --------------------------------
# 4. DISPLAY SALES INFORMATION
# --------------------------------

st.header("Sales Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Units Sold",
    f"{daily_sales.sum():,.0f}"
)

col2.metric(
    "Average Daily Sales",
    f"{daily_sales.mean():,.2f}"
)

col3.metric(
    "Total Records",
    f"{len(df):,}"
)

# --------------------------------
# 5. HISTORICAL SALES GRAPH
# --------------------------------

st.header("Historical Sales Trend")

st.line_chart(daily_sales)

# --------------------------------
# 6. MONTHLY SALES GRAPH
# --------------------------------

st.header("Monthly Sales Analysis")

monthly_sales = daily_sales.resample("ME").sum()

st.bar_chart(monthly_sales)

# --------------------------------
# 7. FUTURE SALES FORECAST
# --------------------------------

st.header("Future Sales Forecast")

forecast_days = st.slider(
    "Select Forecast Days",
    min_value=7,
    max_value=90,
    value=30
)

if st.button("Generate Sales Forecast"):
    model_order = (1, 1, 1)

    if JOBLIB_AVAILABLE and os.path.exists(MODEL_PATH):
        saved_model = joblib.load(MODEL_PATH)
        model_order = getattr(getattr(saved_model, "model", None), "order", (1, 1, 1))
    elif not JOBLIB_AVAILABLE:
        st.info("`joblib` is not installed; using default ARIMA order (1, 1, 1).")
    elif not os.path.exists(MODEL_PATH):
        st.info(f"Saved model not found at `{MODEL_PATH}`; fitting ARIMA with default order (1, 1, 1).")

    final_model = ARIMA(
        daily_sales,
        order=model_order
    ).fit()

    forecast = final_model.forecast(
        steps=forecast_days
    )

    future_dates = pd.date_range(
        start=daily_sales.index.max() + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Units Sold": forecast.values
    })

    st.success("Sales Forecast Generated Successfully!")

    st.subheader("Predicted Sales")

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    st.subheader("Historical vs Future Sales")

    historical = daily_sales.tail(60)

    historical_df = pd.DataFrame({
        "Date": historical.index,
        "Sales": historical.values,
        "Type": "Historical"
    })

    future_df = pd.DataFrame({
        "Date": future_dates,
        "Sales": forecast.values,
        "Type": "Forecast"
    })

    chart_df = pd.concat(
        [historical_df, future_df],
        ignore_index=True
    )

    chart_df = chart_df.pivot(
        index="Date",
        columns="Type",
        values="Sales"
    )

    st.line_chart(chart_df)

    csv = forecast_df.to_csv(index=False)

    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name="sales_forecast.csv",
        mime="text/csv"
    )

# --------------------------------
# 8. FOOTER
# --------------------------------

st.divider()

st.write("Sales Time Series Forecasting Project")

st.write("Developed by Swapna V | Data Science Project")
