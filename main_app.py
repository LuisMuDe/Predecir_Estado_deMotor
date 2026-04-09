import streamlit as st

# ========================
# CONFIG
# ========================
st.set_page_config(layout="wide")
st.title("🧠 Sistema de Monitoreo Inteligente")

# ========================
# MENU
# ========================
st.sidebar.title("📂 Navegación")

app = st.sidebar.radio(
    "Selecciona módulo:",
    [
        "📊 Adquisición de datos",
        "🧠 Entrenamiento ML",
        "🤖 Predicción en tiempo real"
    ]
)

# ========================
# ROUTER
# ========================
if app == "📊 Adquisición de datos":
    import app_motor
    app_motor.run()

elif app == "🧠 Entrenamiento ML":
    import app_training
    app_training.run()

elif app == "🤖 Predicción en tiempo real":
    import app_prediction
    app_prediction.run()