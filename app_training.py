def run():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import joblib

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # ========================
    # CONFIG
    # ========================
    st.set_page_config(layout="wide")
    st.title("🧠 Entrenamiento de Modelo ML")

    # ========================
    # SESSION STATE
    # ========================
    if "model" not in st.session_state:
        st.session_state.model = None

    if "trained" not in st.session_state:
        st.session_state.trained = False

    if "feature_cols" not in st.session_state:
        st.session_state.feature_cols = []

    if "label_col" not in st.session_state:
        st.session_state.label_col = None

    # ========================
    # CARGA DE DATASET
    # ========================
    uploaded_file = st.file_uploader("📂 Sube tu dataset CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Vista del dataset")
        st.dataframe(df.head())

        # ========================
        # CONFIGURACIÓN ML
        # ========================
        st.sidebar.header("⚙️ Configuración ML")

        all_columns = list(df.columns)

        label_col = st.sidebar.selectbox(
            "Columna label",
            all_columns
        )

        feature_cols = st.sidebar.multiselect(
            "Features",
            [col for col in all_columns if col != label_col],
            default=[col for col in all_columns if col not in ["label", "estado"]]
        )

        # Guardar selección
        st.session_state.feature_cols = feature_cols
        st.session_state.label_col = label_col

        # ========================
        # MODELO
        # ========================
        model_type = st.sidebar.selectbox(
            "Modelo",
            ["Random Forest", "KNN"]
        )

        if model_type == "Random Forest":
            n_estimators = st.sidebar.slider("n_estimators", 10, 200, 50)

        if model_type == "KNN":
            n_neighbors = st.sidebar.slider("n_neighbors", 1, 15, 3)

        # ========================
        # ENTRENAMIENTO
        # ========================
        if st.button("🚀 Entrenar modelo"):

            if len(feature_cols) == 0:
                st.error("Selecciona al menos una feature")
            else:
                X = df[feature_cols]
                y = df[label_col]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                if model_type == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators)
                else:
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)

                # 🔥 GUARDAR MODELO EN MEMORIA
                st.session_state.model = model
                st.session_state.trained = True

                st.success(f"Modelo entrenado | Accuracy: {acc:.3f}")

                # ========================
                # IMPORTANCIA FEATURES
                # ========================
                if model_type == "Random Forest":
                    importances = model.feature_importances_

                    feat_imp = pd.DataFrame({
                        "feature": feature_cols,
                        "importance": importances
                    }).sort_values(by="importance", ascending=False)

                    st.subheader("Importancia de features")
                    st.dataframe(feat_imp)

    # ========================
    # GUARDAR MODELO
    # ========================
    st.sidebar.header("💾 Exportar modelo")

    model_name = st.sidebar.text_input("Nombre del modelo", "modelo_motor")

    if st.sidebar.button("Guardar modelo"):

        if st.session_state.trained and st.session_state.model is not None:
            joblib.dump(st.session_state.model, f"{model_name}.joblib")
            st.sidebar.success(f"Modelo guardado como {model_name}.joblib")
        else:
            st.sidebar.error("Primero entrena el modelo")

    # ========================
    # INFO
    # ========================
    if st.session_state.trained:
        st.info("Modelo listo para exportar o usar en predicción en tiempo real")
    
