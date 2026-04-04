import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="XAI-Guard | MalDroid-2020", layout="wide", page_icon="🛡️")

st.markdown("""
<style>
.title { font-size:2rem; font-weight:bold; color:#1F4E79; }
.sub   { color:#595959; font-size:0.95rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">🛡️ XAI-Guard — Android Malware Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">CNN + Enhanced Whale Optimisation Algorithm (EWOA) + XGBoost + SHAP Explainability</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Dataset: CIC-MalDroid-2020 | Binary Classification: Benign vs Malware</p>', unsafe_allow_html=True)
st.divider()

st.sidebar.header("Pipeline")
st.sidebar.markdown("""
**6-Step Architecture:**
1. 📥 Load CIC-MalDroid-2020
2. 🔢 Normalize + SMOTE
3. 🧠 1D-CNN Feature Extraction
4. 🐋 EWOA Feature Selection
5. 🎯 XGBoost Classification
6. 🔍 SHAP — Open the Black Box

**Innovation:**
- EWOA selects minimal features
- SHAP explains every prediction
- No black box — full transparency
""")

tab1, tab2, tab3 = st.tabs(["📊 Results", "🔍 XAI Explanation", "📁 Predict"])

with tab1:
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset",     "CIC-MalDroid-2020")
    c2.metric("Raw Features","470")
    c3.metric("Algorithm",   "EWOA + CNN")
    c4.metric("Explainability","SHAP (XAI)")
    st.divider()

    st.subheader("Training Curves")
    col1, col2 = st.columns(2)
    try:
        col1.image('cnn_training.png',    caption='CNN Training Accuracy & Loss')
        col2.image('ewoa_convergence.png',caption='EWOA Convergence Curve')
    except:
        st.info("Run step2_cnn_ewoa.py first to generate plots.")

    try:
        st.image('confusion_matrix.png', caption='Confusion Matrix', width=450)
    except:
        pass

with tab2:
    st.subheader("🔍 SHAP — Opening the Black Box")
    st.markdown("""
    **Why SHAP?**
    Standard malware detectors are black boxes — they give a verdict but no explanation.
    SHAP shows *which CNN-extracted features* pushed the prediction toward **Malware** or **Benign**,
    making the system transparent and trustworthy for security analysts.
    """)
    try:
        st.image('shap_summary.png',    caption='SHAP Summary — Global Feature Importance')
        st.image('shap_force_plot.png', caption='SHAP Force Plot — Single Prediction Explained')
    except:
        st.info("Run step2_cnn_ewoa.py first.")

with tab3:
    st.subheader("📁 Upload APK Feature CSV for Prediction")
    uploaded = st.file_uploader("Upload CSV (same format as CIC-MalDroid-2020, without Class column)", type=['csv'])

    if uploaded:
        try:
            import tensorflow as tf
            import joblib
            from sklearn.preprocessing import MinMaxScaler

            df_up = pd.read_csv(uploaded)
            if 'Class' in df_up.columns:
                df_up = df_up.drop(columns=['Class'])
            if 'label' in df_up.columns:
                df_up = df_up.drop(columns=['label'])

            st.write(f"Uploaded: {df_up.shape[0]} samples, {df_up.shape[1]} features")

            df_num = df_up.select_dtypes(include=[np.number]).fillna(0)
            scaler = MinMaxScaler()
            X_new  = scaler.fit_transform(df_num)
            X_cnn  = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

            cnn_model = tf.keras.models.load_model('cnn_model.h5')
            lat_model = tf.keras.Model(
                inputs=cnn_model.input,
                outputs=cnn_model.get_layer('latent_features').output
            )
            X_lat = lat_model.predict(X_cnn, verbose=0)

            sel = np.load('selected_indices.npy')
            # Clip indices to available range
            sel = sel[sel < X_lat.shape[1]]
            X_sel = X_lat[:, sel]

            clf   = joblib.load('xgb_model.pkl')
            preds = clf.predict(X_sel)
            probs = clf.predict_proba(X_sel)[:, 1]

            results = pd.DataFrame({
                'Prediction': ['🔴 Malware' if p == 1 else '🟢 Benign' for p in preds],
                'Malware Probability': [f"{p*100:.1f}%" for p in probs]
            })
            st.dataframe(results)
            st.metric("Malware Detected", f"{np.sum(preds==1)} / {len(preds)}")

        except Exception as e:
            st.error(f"Error: {e}")
            st.warning("Make sure step2_cnn_ewoa.py has been run first to generate model files.")

st.divider()
st.caption("Built by Punna Bhargavi | Mini Project 3-2 | CNN + EWOA + SHAP | CIC-MalDroid-2020")