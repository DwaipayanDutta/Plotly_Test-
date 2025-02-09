import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import os

cwd = os.getcwd()
# Load the dataset (make sure the CSV file is in the same folder)
df = pd.read_csv("OneDrive/Desktop/Plotly_app/archive/credit_card_transactions.csv")

# Preprocess the data
# Convert 'trans_date_trans_time' to datetime and extract features
df["trans_date_trans_time"] = pd.to_datetime(
    df["trans_date_trans_time"], errors="coerce"
)
df["trans_year"] = df["trans_date_trans_time"].dt.year
df["trans_month"] = df["trans_date_trans_time"].dt.month
df["trans_day"] = df["trans_date_trans_time"].dt.day
df["trans_hour"] = df["trans_date_trans_time"].dt.hour

# Drop original timestamp column and any unnecessary columns
df.drop(
    columns=[
        "trans_date_trans_time",
        "cc_num",
        "trans_num",
        "first",
        "last",
        "street",
        "city",
        "state",
        "zip",
    ],
    inplace=True,
)

# Handle categorical variables using one-hot encoding or label encoding
df = pd.get_dummies(df, drop_first=True)

# Ensure target variable is integer type and check for NaN values in features
df["is_fraud"] = df["is_fraud"].astype(int)

# Check for NaN values and drop them if necessary
df.dropna(inplace=True)

# Clean feature names to remove special characters and spaces
df.columns = (
    df.columns.str.replace(" ", "_").str.replace("-", "_").str.replace(".", "_")
)

# EDA Section: Summary Statistics and Visualizations

# Summary statistics
summary_stats = df.describe()

# Distribution of target variable (is_fraud)
fraud_counts = df["is_fraud"].value_counts().reset_index()
fraud_counts.columns = ["is_fraud", "count"]  # Rename columns for clarity

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Credit Card Fraud Detection"),
        html.Div(
            [
                html.H2("Exploratory Data Analysis"),
                dcc.Graph(
                    id="fraud-distribution",
                    figure=px.pie(
                        fraud_counts,
                        names="is_fraud",
                        values="count",
                        title="Distribution of Fraudulent vs Non-Fraudulent Transactions",
                    ),
                ),
                html.Div(
                    [
                        html.H3("Summary Statistics"),
                        dcc.Graph(
                            id="summary-stats",
                            figure=go.Figure(
                                data=[
                                    go.Table(
                                        header=dict(
                                            values=list(summary_stats.columns),
                                            fill_color="paleturquoise",
                                            align="left",
                                        ),
                                        cells=dict(
                                            values=[
                                                summary_stats[col]
                                                for col in summary_stats.columns
                                            ],
                                            fill_color="lavender",
                                            align="left",
                                        ),
                                    )
                                ]
                            ),
                        ),
                    ]
                ),
            ]
        ),
        html.Div(
            [
                html.H2("Model Performance"),
                dcc.Dropdown(
                    id="metric-dropdown",
                    options=[
                        {"label": "Mean F1 Score", "value": "Mean F1 Score"},
                        {"label": "Mean Precision", "value": "Mean Precision"},
                        {"label": "Mean Recall", "value": "Mean Recall"},
                    ],
                    value="Mean F1 Score",
                ),
                dcc.Graph(id="performance-graph"),
            ]
        ),
    ]
)


@app.callback(Output("performance-graph", "figure"), Input("metric-dropdown", "value"))
def update_graph(selected_metric):
    # Train-test split for model training and evaluation
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train models and evaluate performance
    models = {
        "CatBoost": CatBoostClassifier(silent=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    }

    results = []

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        results.append(
            {
                "Model": model_name,
                "Mean F1 Score": f1,
                "Mean Precision": precision,
                "Mean Recall": recall,
            }
        )

    # Convert results to DataFrame for visualization
    results_df = pd.DataFrame(results)

    fig = px.bar(
        results_df,
        x="Model",
        y=selected_metric,
        title=f"Model Performance: {selected_metric}",
        labels={"Model": "Machine Learning Model", selected_metric: selected_metric},
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
