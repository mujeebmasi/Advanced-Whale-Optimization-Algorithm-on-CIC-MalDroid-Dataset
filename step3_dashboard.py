import io
import warnings

import joblib
import numpy as np
import pandas as pd
import requests
import shap
import streamlit as st
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

ARTIFACT_XGB = "xgb_model.pkl"
ARTIFACT_PCA = "pca_model.pkl"
ARTIFACT_IDX = "selected_indices.npy"

MALWARE_REASON_LIBRARY = [
    "The app is trying to access sensitive phone data.",
    "The app may be communicating with unknown internet servers.",
    "The app may run hidden background processes.",
    "The app may try to collect private user information.",
    "The app may be requesting risky permissions that are not needed.",
    "The app may be attempting suspicious file system activity.",
    "The app may be attempting to stay active without user awareness.",
    "The app may be showing behavior linked to data exfiltration.",
    "The app may be interacting with device identifiers unusually.",
    "The app may be using stealthy behavior patterns.",
]


@st.cache_resource
def load_artifacts():
    model = joblib.load(ARTIFACT_XGB)
    pca = joblib.load(ARTIFACT_PCA)
    selected_indices = np.load(ARTIFACT_IDX)

    selected_indices = np.asarray(selected_indices, dtype=int)
    selected_indices = np.unique(selected_indices)

    if selected_indices.size == 0:
        raise ValueError("selected_indices.npy is empty.")

    if np.any(selected_indices < 0):
        raise ValueError("selected_indices.npy contains negative indices.")

    return model, pca, selected_indices


def parse_api_payload(payload):
    if isinstance(payload, list):
        return pd.DataFrame(payload)

    if isinstance(payload, dict):
        for key in ["data", "records", "results", "items"]:
            value = payload.get(key)
            if isinstance(value, list):
                return pd.DataFrame(value)
            if isinstance(value, dict):
                return pd.DataFrame([value])
        return pd.DataFrame([payload])

    raise ValueError("Unsupported API response format.")


def fetch_from_api(api_url):
    response = requests.get(api_url, timeout=20)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()
    if "application/json" in content_type:
        return parse_api_payload(response.json())

    try:
        return pd.read_csv(io.StringIO(response.text))
    except Exception as exc:
        raise ValueError("API did not return valid JSON or CSV data.") from exc


def preprocess_input(df, pca):
    if df.empty:
        raise ValueError("Input data is empty.")

    data = df.copy()
    for col in ["label", "Label", "class", "Class", "target", "Target"]:
        if col in data.columns:
            data = data.drop(columns=[col])

    numeric = data.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(0.0)

    expected = getattr(pca, "n_features_in_", None)
    if expected is None:
        raise ValueError("PCA artifact does not expose expected feature count.")

    if numeric.shape[1] != expected:
        raise ValueError(
            f"Incorrect feature count. Expected {expected} numeric features, got {numeric.shape[1]}."
        )

    return numeric.to_numpy(dtype=float)


def extract_ground_truth_labels(df):
    if "label" not in df.columns:
        return None

    label_series = df["label"].copy()
    if label_series.isna().any():
        return None

    if pd.api.types.is_numeric_dtype(label_series):
        labels = label_series.astype(int).to_numpy()
    else:
        normalized = label_series.astype(str).str.strip().str.lower()
        mapping = {
            "benign": 0,
            "safe": 0,
            "normal": 0,
            "malware": 1,
            "malicious": 1,
            "attack": 1,
        }
        if not normalized.isin(mapping.keys()).all():
            return None
        labels = normalized.map(mapping).astype(int).to_numpy()

    if not np.isin(labels, [0, 1]).all():
        return None

    return labels


def run_inference(model, pca, selected_indices, x_raw):
    x_pca = pca.transform(x_raw)

    if np.max(selected_indices) >= x_pca.shape[1]:
        raise ValueError("selected_indices.npy contains indices outside PCA output range.")

    if selected_indices.size > 10:
        raise ValueError(
            f"EWOA selected {selected_indices.size} features. Expected 10 or fewer."
        )

    x_selected = x_pca[:, selected_indices]
    predictions = model.predict(x_selected)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_selected)[:, 1]
    else:
        probabilities = np.full(shape=(x_selected.shape[0],), fill_value=np.nan)

    return x_selected, predictions, probabilities


def compute_shap(model, x_selected, selected_indices):
    explain_count = min(100, x_selected.shape[0])
    x_explain = x_selected[:explain_count]

    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(x_explain)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]

        return x_explain, values
    except Exception:
        background_count = min(100, x_selected.shape[0])
        background = shap.sample(x_selected, background_count, random_state=42)
        masker = shap.maskers.Independent(background)
        explainer = shap.Explainer(
            lambda data: model.predict_proba(data)[:, 1],
            masker,
            algorithm="permutation",
        )
        shap_exp = explainer(x_explain, max_evals=2 * x_explain.shape[1] + 1)
        values = shap_exp.values
        return x_explain, values


def render_shap_text_explanations(first_prediction, shap_values):
    first_row_values = shap_values[0]
    top_indices = np.argsort(np.abs(first_row_values))[::-1][:5]

    st.write("Explainable AI Result:")
    if first_prediction == 1:
        st.write("The application may be malware because:")
    else:
        st.write("The application appears benign because major malware-like behaviors are limited:")

    for idx in top_indices:
        reason = MALWARE_REASON_LIBRARY[idx % len(MALWARE_REASON_LIBRARY)]
        if first_prediction == 1:
            st.write(f"• {reason}")
        else:
            if first_row_values[idx] < 0:
                st.write(f"• Low evidence for this risk: {reason}")
            else:
                st.write(f"• Mild signal observed: {reason}")


def render_home_page():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 15% 10%, #ffffff 0%, #eef4ff 40%, #dbe9ff 100%);
        }
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
        .home-wrap {
            min-height: 74vh;
            padding: 3.2rem 2.5rem;
            border-radius: 24px;
            background: linear-gradient(120deg, #2563eb 0%, #7c3aed 45%, #06b6d4 100%);
            color: #ffffff;
            box-shadow: 0 20px 45px rgba(37, 99, 235, 0.28);
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 1rem;
        }
        .home-title {
            font-size: 2.8rem;
            font-weight: 800;
            letter-spacing: 0.2px;
            margin-bottom: 0.2rem;
        }
        .home-tag {
            font-size: 1rem;
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            width: fit-content;
        }
        .home-sub {
            font-size: 1.12rem;
            line-height: 1.85;
            opacity: 0.98;
            max-width: 960px;
            margin-top: 0.5rem;
        }
        .home-cards {
            display: grid;
            grid-template-columns: repeat(3, minmax(180px, 1fr));
            gap: 0.8rem;
            margin-top: 0.9rem;
            max-width: 960px;
        }
        .home-card {
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.34);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            font-size: 0.95rem;
            line-height: 1.35;
        }
        </style>
        <div class="home-wrap">
            <div class="home-tag">Secure AI Malware Screening</div>
            <div class="home-title">Android Malware Detection with Explainable AI</div>
            <div class="home-sub">
                Protect your Android security workflow with a bright, intelligent detection dashboard.<br>
                This project combines PCA, EWOA feature selection, XGBoost prediction, and user-friendly explainable AI.<br>
                Upload a CSV or fetch live records from an API to analyze apps in seconds.
            </div>
            <div class="home-cards">
                <div class="home-card"><b>Fast Screening</b><br>Analyze uploaded or live API feature data instantly.</div>
                <div class="home-card"><b>Explainable Results</b><br>Understand why an app is risky with plain-language reasons.</div>
                <div class="home-card"><b>Model Pipeline</b><br>PCA + EWOA + XGBoost optimized for malware detection.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Open Malware Detection Dashboard", type="primary", use_container_width=True):
        st.session_state["show_detection_page"] = True
        st.rerun()


def main():
    st.set_page_config(page_title="Android Malware Detection with Explainable AI", layout="wide")

    if "show_detection_page" not in st.session_state:
        st.session_state["show_detection_page"] = False

    if not st.session_state["show_detection_page"]:
        render_home_page()
        return

    st.title("Android Malware Detection with Explainable AI")
    st.caption("PCA + EWOA + XGBoost + SHAP")

    try:
        model, pca, selected_indices = load_artifacts()
    except Exception as exc:
        st.error(f"Failed to load model artifacts: {exc}")
        st.stop()

    st.write(f"Loaded artifacts: `{ARTIFACT_XGB}`, `{ARTIFACT_PCA}`, `{ARTIFACT_IDX}`")
    st.write(f"EWOA selected feature count: {selected_indices.size}")

    if selected_indices.size > 10:
        st.error("EWOA selected feature count is greater than 10. Retrain or replace artifacts.")
        st.stop()

    st.subheader("Input Data")
    input_mode = st.radio(
        "Choose input source",
        ["Upload CSV", "Fetch from API"],
        horizontal=True,
    )

    input_df = None

    if input_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV feature data", type=["csv"])
        if uploaded is not None:
            try:
                input_df = pd.read_csv(uploaded)
                st.success(f"Loaded CSV with shape: {input_df.shape}")
                st.dataframe(input_df.head(10), use_container_width=True)
            except Exception as exc:
                st.error(f"Invalid CSV file: {exc}")
    else:
        api_url = st.text_input("REST API endpoint URL", placeholder="https://example.com/features")
        if st.button("Fetch Data from API", type="primary"):
            if not api_url.strip():
                st.warning("Please enter an API URL.")
            else:
                try:
                    with st.spinner("Fetching data from API..."):
                        input_df = fetch_from_api(api_url.strip())
                    st.success(f"Fetched API data with shape: {input_df.shape}")
                    st.dataframe(input_df.head(10), use_container_width=True)
                except requests.RequestException as exc:
                    st.error(f"API request failed: {exc}")
                except ValueError as exc:
                    st.error(f"API data error: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected API error: {exc}")

    if input_df is None:
        st.info("Provide data using one of the input options to run detection.")
        return

    true_labels = extract_ground_truth_labels(input_df)

    try:
        x_raw = preprocess_input(input_df, pca)
        x_selected, predictions, probabilities = run_inference(model, pca, selected_indices, x_raw)
    except Exception as exc:
        st.error(f"Prediction pipeline failed: {exc}")
        return

    st.subheader("Prediction Results")

    first_pred = int(predictions[0])
    first_label = "Malware Detected" if first_pred == 1 else "Benign Application"
    first_color = "#D32F2F" if first_pred == 1 else "#2E7D32"
    st.markdown(
        f"<h3 style='color:{first_color};'>{first_label}</h3>",
        unsafe_allow_html=True,
    )

    results = pd.DataFrame(
        {
            "Prediction": np.where(predictions == 1, "Malware Detected", "Benign Application"),
            "Malware Probability": np.round(probabilities, 4),
        }
    )
    st.dataframe(results, use_container_width=True)

    malware_count = int(np.sum(predictions == 1))
    benign_count = int(np.sum(predictions == 0))
    col1, col2 = st.columns(2)
    col1.metric("Malware Samples", malware_count)
    col2.metric("Benign Samples", benign_count)

    st.subheader("SHAP Explainability")
    try:
        _, shap_values = compute_shap(model, x_selected, selected_indices)
        render_shap_text_explanations(first_pred, shap_values)
    except Exception as exc:
        st.error(f"SHAP explanation failed: {exc}")

    st.subheader("Final Accuracy")
    if true_labels is not None and len(true_labels) == len(predictions):
        accuracy = accuracy_score(true_labels, predictions) * 100
        st.write(f"Accuracy on provided data: {accuracy:.2f}%")

        true_labels_int = true_labels.astype(int)
        predictions_int = predictions.astype(int)
        tp = int(np.sum((true_labels_int == 1) & (predictions_int == 1)))
        tn = int(np.sum((true_labels_int == 0) & (predictions_int == 0)))
        fp = int(np.sum((true_labels_int == 0) & (predictions_int == 1)))
        fn = int(np.sum((true_labels_int == 1) & (predictions_int == 0)))

        st.write("Confusion Matrix Summary:")
        confusion_df = pd.DataFrame(
            [{"TP": tp, "TN": tn, "FP": fp, "FN": fn}]
        )
        st.dataframe(confusion_df, use_container_width=True)
    else:
        st.info("Live prediction mode — accuracy cannot be calculated because the true label is unknown.")


if __name__ == "__main__":
    main()
