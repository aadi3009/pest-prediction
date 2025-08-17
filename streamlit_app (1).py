"""
A Streamlit dashboard for agricultural field deployment
======================================================

This application provides an interactive interface for exploring a cotton pest
dataset and comparing a suite of machine‑learning models for predicting
infestations.  Users can browse the raw data, perform exploratory analysis
through interactive plots, review model performance metrics, inspect feature
importance scores and generate predictions for future weeks by entering
weather parameters.  The models implemented include Linear Regression,
Support Vector Regression and Random Forest Regressor.  Performance
statistics for additional deep learning models (CNN and time‑series CNN)
reported in the associated project report are displayed for comparison.

The code is organised into logical sections with cached functions for
efficient data loading and model training.  Plotly is used for all charts to
provide interactive visualisations within the Streamlit framework.
"""

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    import streamlit as st
except ImportError as e:
    raise RuntimeError(
        "Streamlit is required to run this application. "
        "Please install it in your environment before executing this script."
    ) from e


###############################################################################
# Data loading and preprocessing
###############################################################################

@st.cache_data(show_spinner=False)
def load_dataset(path: str) -> pd.DataFrame:
    """Load the cotton dataset from a CSV file and preprocess it.

    Missing values in the ``Pest Value`` column are imputed with zero and
    a log‑transformed version of the target is added as ``Log Pest Value``.

    Parameters
    ----------
    path: str
        Path to the CSV file containing the data.

    Returns
    -------
    pd.DataFrame
        A processed DataFrame ready for analysis and modelling.
    """
    data = pd.read_csv(path)
    # Fill missing pest values with 0 for weeks when no crop was present
    data["Pest Value"].fillna(0, inplace=True)
    # Log transform to stabilise variance and handle zeros
    data["Log Pest Value"] = np.log(data["Pest Value"] + 1)
    return data


@st.cache_resource(show_spinner=False)
def split_features_labels(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract the feature matrix and target vector for model training.

    Only meteorological parameters are used as features; the log‑transformed
    pest value is used as the target.

    Parameters
    ----------
    data: pd.DataFrame
        Processed dataset returned by ``load_dataset``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        ``X``: feature DataFrame
        ``y``: target Series
    """
    feature_cols = [
        "MaxT(°C)",
        "MinT(°C)",
        "RH1(%)",
        "RH2(%)",
        "RF(mm)",
        "WS(kmph)",
        "SSH(hrs)",
        "EVP(mm)",
    ]
    X = data[feature_cols].copy()
    y = data["Log Pest Value"]
    return X, y


###############################################################################
# Model training
###############################################################################

@st.cache_resource(show_spinner=True)
def train_models(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict[str, any]]:
    """Train multiple regression models and compute performance metrics.

    Three models are trained: Linear Regression, Support Vector Regression
    with an RBF kernel and Random Forest Regressor.  Each model is trained
    using an 80/20 train/test split.  A ``StandardScaler`` is applied to
    features for SVR.  Performance metrics (MAE, RMSE, R²) are calculated on
    the held‑out test set.

    Parameters
    ----------
    X: pd.DataFrame
        Feature matrix.
    y: pd.Series
        Target vector.
    test_size: float, optional
        Fraction of the data to reserve for testing.  Default is 0.2.
    random_state: int, optional
        Random seed for reproducibility.  Default is 42.

    Returns
    -------
    Dict[str, Dict[str, any]]
        A dictionary keyed by model name containing the fitted model,
        predictions and performance metrics.
    """
    results: Dict[str, Dict[str, any]] = {}
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Linear Regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    y_pred_lin = lin_model.predict(X_test)
    results["Linear Regression"] = {
        "model": lin_model,
        "predictions": y_pred_lin,
        "metrics": _compute_metrics(y_test, y_pred_lin),
    }

    # Support Vector Regression with scaling
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    svr = SVR(kernel="rbf", C=10, gamma=0.1)
    svr.fit(X_train_s, y_train)
    y_pred_svr = svr.predict(X_test_s)
    results["Support Vector Regression"] = {
        "model": svr,
        "scaler": scaler,
        "predictions": y_pred_svr,
        "metrics": _compute_metrics(y_test, y_pred_svr),
    }

    # Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results["Random Forest Regressor"] = {
        "model": rf,
        "predictions": y_pred_rf,
        "metrics": _compute_metrics(y_test, y_pred_rf),
        "feature_importances": rf.feature_importances_,
    }

    return results


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute common regression metrics.

    Metrics include Mean Absolute Error (MAE), Root Mean Squared Error (RMSE)
    and the coefficient of determination (R²).

    Parameters
    ----------
    y_true: pd.Series
        Actual target values.
    y_pred: np.ndarray
        Predicted values from the model.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the computed metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R^2": r2}


###############################################################################
# Visualisations
###############################################################################

def correlation_heatmap(data: pd.DataFrame) -> go.Figure:
    """Generate an interactive heatmap of absolute Spearman correlations.

    Parameters
    ----------
    data: pd.DataFrame
        Processed dataset containing the log‑transformed target.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure representing the heatmap.
    """
    cols = [
        "Log Pest Value",
        "MaxT(°C)",
        "MinT(°C)",
        "RH1(%)",
        "RH2(%)",
        "RF(mm)",
        "WS(kmph)",
        "SSH(hrs)",
        "EVP(mm)",
    ]
    corr = data[cols].corr(method="spearman").abs()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=0,
            zmax=1,
            colorbar=dict(title="|ρ|"),
        )
    )
    fig.update_layout(
        title="Spearman Correlation Heatmap (absolute values)",
        xaxis_title="Variables",
        yaxis_title="Variables",
        width=600,
        height=500,
    )
    return fig


def pest_trends_line_chart(data: pd.DataFrame, use_log: bool = True) -> go.Figure:
    """Create an interactive line chart showing pest trends across weeks for each year.

    Parameters
    ----------
    data: pd.DataFrame
        Dataset including ``Observation Year``, ``Standard Week`` and target.
    use_log: bool, optional
        Whether to plot the log‑transformed pest value.  If ``False``, the
        original ``Pest Value`` will be used.  Default is ``True``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive line chart.
    """
    y_col = "Log Pest Value" if use_log else "Pest Value"
    fig = px.line(
        data,
        x="Standard Week",
        y=y_col,
        color="Observation Year",
        labels={"Observation Year": "Year", "Standard Week": "Week", y_col: y_col},
        title=f"{'Log ' if use_log else ''}Pest Value Trends by Week",
    )
    fig.update_layout(height=500, width=700)
    return fig


def feature_importance_bar(importances: np.ndarray, feature_names: list) -> go.Figure:
    """Render a horizontal bar chart of feature importances.

    Parameters
    ----------
    importances: np.ndarray
        Importance values from the Random Forest model.
    feature_names: list
        Names corresponding to the importance scores.

    Returns
    -------
    plotly.graph_objects.Figure
        Horizontal bar chart figure.
    """
    sorted_idx = np.argsort(importances)
    sorted_importances = importances[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    fig = go.Figure(
        go.Bar(
            x=sorted_importances,
            y=sorted_names,
            orientation="h",
            marker=dict(color="skyblue"),
        )
    )
    fig.update_layout(
        title="Feature Importance (Random Forest)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        width=600,
        margin=dict(l=80, r=20, t=50, b=50),
    )
    return fig


###############################################################################
# Application pages
###############################################################################

def overview_page(data: pd.DataFrame) -> None:
    """Display an overview of the dataset.

    This page shows a few rows from the dataset, summarises its shape and
    provides a table describing each column in plain language.  A basic
    distribution plot of the original pest values is also included.
    """
    st.header("Dataset Overview")
    st.markdown(
        "This dataset contains weekly observations of weather parameters and "
        "pest infestation values collected in Coimbatore between 2001 and 2009. "
        "Missing pest values for weeks before sowing have been imputed with zeros, "
        "and a log transformation has been applied to stabilise variance in the target."
    )

    # Show data sample
    st.subheader("Sample records")
    st.dataframe(data.head(10))

    # Show basic stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of records", f"{len(data):,}")
    col2.metric("Number of features", f"{data.shape[1] - 3}")
    col3.metric(
        "Year span",
        f"{int(data['Observation Year'].min())} – {int(data['Observation Year'].max())}",
    )

    # Column description table
    st.subheader("Column descriptions")
    col_desc = pd.DataFrame(
        {
            "Column": [
                "Observation Year",
                "Standard Week",
                "Pest Value",
                "MaxT(°C)",
                "MinT(°C)",
                "RH1(%)",
                "RH2(%)",
                "RF(mm)",
                "WS(kmph)",
                "SSH(hrs)",
                "EVP(mm)",
            ],
            "Description": [
                "Calendar year of the observation.",
                "ISO standard week number (1–52).",
                "Measured pest population (adults per trap).",
                "Maximum daily temperature (°C).",
                "Minimum daily temperature (°C).",
                "Morning relative humidity (%).",
                "Evening relative humidity (%).",
                "Total rainfall during the week (mm).",
                "Mean wind speed (km/h).",
                "Hours of bright sunshine.",
                "Evaporation (mm).",
            ],
        }
    )
    st.dataframe(col_desc)

    # Distribution of pest values
    st.subheader("Pest value distribution")
    hist_fig = px.histogram(
        data,
        x="Pest Value",
        nbins=30,
        title="Distribution of raw pest values",
        labels={"Pest Value": "Pest Value"},
    )
    st.plotly_chart(hist_fig, use_container_width=True)


def exploratory_analysis_page(data: pd.DataFrame) -> None:
    """Provide exploratory data analysis through interactive plots."""
    st.header("Exploratory Analysis")
    st.markdown(
        "Investigate relationships between weather parameters and pest infestation. "
        "Use the controls below to switch between the log‑transformed and original "
        "pest values."
    )

    use_log = st.radio(
        "Value scale", ["Log scale", "Original scale"], index=0, horizontal=True
    )
    st.plotly_chart(correlation_heatmap(data), use_container_width=True)
    st.plotly_chart(
        pest_trends_line_chart(data, use_log=(use_log == "Log scale")),
        use_container_width=True,
    )

    # Pairwise scatter matrix
    st.subheader("Pairwise relationships")
    cols = [
        "Log Pest Value" if use_log == "Log scale" else "Pest Value",
        "MaxT(°C)",
        "MinT(°C)",
        "RH1(%)",
        "RH2(%)",
        "RF(mm)",
        "WS(kmph)",
        "SSH(hrs)",
        "EVP(mm)",
    ]
    scatter_fig = px.scatter_matrix(
        data,
        dimensions=cols,
        color="Observation Year",
        title="Scatter matrix of selected variables",
        height=700,
    )
    # Adjust axis labels for readability
    scatter_fig.update_layout(
        showlegend=False,
        dragmode="select",
    )
    st.plotly_chart(scatter_fig, use_container_width=True)


def model_comparison_page(data: pd.DataFrame) -> None:
    """Train and compare the performance of different regression models."""
    st.header("Model Comparison")
    st.markdown(
        "Three models are trained on the meteorological parameters to predict the log‑transformed pest value. "
        "The table below summarises their performance.  Additional deep‑learning models from the project report "
        "are included for context."
    )
    X, y = split_features_labels(data)
    results = train_models(X, y)

    # Build a DataFrame summarising the metrics
    rows = []
    for name, res in results.items():
        metrics = res["metrics"]
        rows.append(
            {
                "Model": name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R^2": metrics["R^2"],
            }
        )

    # Append reported metrics from the PDF report
    reported_metrics = [
        {
            "Model": "CNN (weather parameters)",
            "MAE": 0.6542,
            "RMSE": 1.1141,
            "R^2": 0.6869,
        },
        {
            "Model": "Time Series CNN",
            "MAE": 0.0564,
            "RMSE": 0.0901,
            "R^2": 0.8643,
        },
    ]
    rows.extend(reported_metrics)
    metrics_df = pd.DataFrame(rows).sort_values(by="R^2", ascending=False)
    st.dataframe(
        metrics_df.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R^2": "{:.4f}"})
    )

    # Feature importance for Random Forest
    rf_importances = results["Random Forest Regressor"].get("feature_importances")
    if rf_importances is not None:
        st.subheader("Random Forest feature importance")
        fig = feature_importance_bar(rf_importances, X.columns.tolist())
        st.plotly_chart(fig, use_container_width=True)

    # Detailed explanation section
    with st.expander("Interpretation of results"):
        st.markdown(
            "* **Random Forest and Time Series CNN models perform best.** The "
            "Random Forest achieves the smallest MAE and RMSE among the traditional "
            "models trained on weather parameters, while the Time Series CNN—trained "
            "solely on past pest values—yields the highest R² score.\n"
            "* **Humidity plays a significant role.** Feature importance from the Random "
            "Forest model highlights relative humidity (RH1 and RH2) alongside maximum "
            "temperature and evaporation as key drivers of pest proliferation.  This "
            "observation aligns with agronomic knowledge that high humidity fosters "
            "favourable conditions for pink bollworm growth.\n"
            "* **Linear models underperform.** Linear Regression and Support Vector Regression "
            "models struggle to capture the complex non‑linear relationships between weather "
            "variables and pest infestation, resulting in higher error metrics."
        )


def prediction_page(data: pd.DataFrame) -> None:
    """Provide a tool for forecasting pest values based on user input."""
    st.header("Pest Infestation Predictor")
    st.markdown(
        "Enter current or forecasted weather conditions to estimate the expected level of "
        "pink bollworm infestation for the upcoming week.  Predictions are generated using "
        "one of three machine‑learning models trained on historical data.  Results are "
        "displayed both in the log scale (used for modelling) and converted back to the "
        "original scale for ease of interpretation."
    )

    # Load features and trained models
    X, y = split_features_labels(data)
    results = train_models(X, y)

    # Sidebar for model selection
    model_name = st.selectbox(
        "Choose a model for prediction:", list(results.keys()), index=2
    )
    model_info = results[model_name]
    model = model_info["model"]
    scaler = model_info.get("scaler")  # only present for SVR

    # Input widgets for weather parameters with sensible defaults
    st.subheader("Weather inputs")
    c1, c2, c3, c4 = st.columns(4)
    max_temp = c1.number_input(
        "Max temperature (°C)", value=float(data["MaxT(°C)"].median())
    )
    min_temp = c2.number_input(
        "Min temperature (°C)", value=float(data["MinT(°C)"].median())
    )
    rh1 = c3.number_input(
        "Morning relative humidity (%)", value=float(data["RH1(%)"].median())
    )
    rh2 = c4.number_input(
        "Evening relative humidity (%)", value=float(data["RH2(%)"].median())
    )
    c5, c6, c7, c8 = st.columns(4)
    rf = c5.number_input(
        "Rainfall (mm)", value=float(data["RF(mm)"].median())
    )
    ws = c6.number_input(
        "Wind speed (km/h)", value=float(data["WS(kmph)"].median())
    )
    ssh = c7.number_input(
        "Sunshine hours", value=float(data["SSH(hrs)"].median())
    )
    evp = c8.number_input(
        "Evaporation (mm)", value=float(data["EVP(mm)"].median())
    )

    if st.button("Predict pest value"):
        # Assemble input into DataFrame
        input_df = pd.DataFrame(
            {
                "MaxT(°C)": [max_temp],
                "MinT(°C)": [min_temp],
                "RH1(%)": [rh1],
                "RH2(%)": [rh2],
                "RF(mm)": [rf],
                "WS(kmph)": [ws],
                "SSH(hrs)": [ssh],
                "EVP(mm)": [evp],
            }
        )
        # Apply scaler if needed
        if scaler is not None:
            input_processed = scaler.transform(input_df)
        else:
            input_processed = input_df
        # Predict log value
        log_pred = model.predict(input_processed)[0]
        pest_pred = math.exp(log_pred) - 1
        st.success(
            f"Predicted log pest value: {log_pred:.3f}\n"
            f"Predicted pest value (original scale): {pest_pred:.3f}"
        )
        # Provide simple interpretation
        if pest_pred < 1:
            level = "Low"
            advice = (
                "Infestation risk appears minimal. Monitor your crop but immediate "
                "intervention is not necessary."
            )
        elif pest_pred < 3:
            level = "Moderate"
            advice = (
                "Infestation risk is moderate. Consider setting pheromone traps and "
                "scouting regularly for bollworm activity."
            )
        else:
            level = "High"
            advice = (
                "Infestation risk is high. Consult local extension guidelines for "
                "appropriate control measures, which may include timely insecticide "
                "applications and maintaining field hygiene."
            )
        st.write(f"**Risk level:** {level}")
        st.info(advice)


###############################################################################
# Main application entry point
###############################################################################

def main() -> None:
    st.set_page_config(
        page_title="Cotton Pest Prediction Dashboard",
        page_icon="🐛",
        layout="wide",
    )
    st.title("Cotton Pest Prediction Dashboard")
    data = load_dataset("cotton_dataset.csv")

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigate", ["Overview", "Exploratory Analysis", "Model Comparison", "Predictor"], index=0
    )
    if page == "Overview":
        overview_page(data)
    elif page == "Exploratory Analysis":
        exploratory_analysis_page(data)
    elif page == "Model Comparison":
        model_comparison_page(data)
    elif page == "Predictor":
        prediction_page(data)


if __name__ == "__main__":
    main()