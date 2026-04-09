def run():
    import streamlit as st
    import smbus2
    import numpy as np
    import pandas as pd
    import joblib
    import time
    import threading

    # ========================
    # CONFIG
    # ========================
    st.set_page_config(layout="wide")
    st.title("🤖 Sistema de Monitoreo de Motor DC")

    # ========================
    # GLOBAL STATE (PERSISTENTE)
    # ========================
    if "global_state" not in st.session_state:
        st.session_state.global_state = {
            "running": False,
            "mag_buffer": [],
            "win_buffer": [],
            "pred_history": [],
            "pred_time": [],
            "pred_binary": [],
            "last_prediction": "..."
        }

    GLOBAL_STATE = st.session_state.global_state

    if "lock" not in st.session_state:
        st.session_state.lock = threading.Lock()

    LOCK = st.session_state.lock

    if "thread" not in st.session_state:
        st.session_state.thread = None

    FAIL_STATES = ["Normal", "Falla"]

    # ========================
    # I2C INIT (UNA VEZ)
    # ========================
    if "i2c_bus" not in st.session_state:
        st.session_state.i2c_bus = smbus2.SMBus(1)

    bus = st.session_state.i2c_bus

    ADXL_ADDR = 0x53
    POWER_CTL = 0x2D
    DATA_FORMAT = 0x31
    DATAX0 = 0x32
    MLX_ADDR = 0x5A

    if "sensor_initialized" not in st.session_state:
        try:
            bus.write_byte_data(ADXL_ADDR, POWER_CTL, 0x08)
            bus.write_byte_data(ADXL_ADDR, DATA_FORMAT, 0x08)
            st.session_state.sensor_initialized = True
        except Exception as e:
            st.error(f"Error inicializando sensor: {e}")

    # ========================
    # FUNCIONES SEGURAS
    # ========================
    def read_axis(addr):
        low = bus.read_byte_data(ADXL_ADDR, addr)
        high = bus.read_byte_data(ADXL_ADDR, addr + 1)
        value = (high << 8) | low
        if value & (1 << 15):
            value -= (1 << 16)
        return value * 0.004

    def safe_read_axis(addr):
        try:
            return read_axis(addr)
        except:
            return 0.0

    def read_temp():
        data = bus.read_word_data(MLX_ADDR, 0x07)
        return (data * 0.02) - 273.15

    def safe_read_temp():
        try:
            return read_temp()
        except:
            return 25.0

    # ========================
    # SIDEBAR
    # ========================
    st.sidebar.header("📦 Modelo")

    uploaded_model = st.sidebar.file_uploader("Cargar modelo (.joblib)", type=["joblib"])

    if uploaded_model is not None:
        data = joblib.load(uploaded_model)
        if isinstance(data, dict):
            model = data["model"]
            feature_order = data["features"]
        else:
            model = data
            feature_order = ["rms","freq","temp","voltaje","corriente","fs","window"]

        st.sidebar.success("Modelo cargado")
        st.sidebar.write("Features:", feature_order)
    else:
        model = None
        feature_order = ["rms","freq","temp","voltaje","corriente","fs","window"]

    FS = st.sidebar.number_input("Frecuencia de muestreo (Hz)", 1, 200, 10)
    WINDOW_SIZE = st.sidebar.number_input("Tamaño de ventana", 16, 512, 64)

    voltaje = st.sidebar.number_input("Voltaje (V)", 0.0, 100.0, 12.0)
    corriente = st.sidebar.number_input("Corriente (A)", 0.0, 50.0, 1.5)

    # ========================
    # WORKER THREAD
    # ========================
    def worker_loop(model, feature_order, FS, WINDOW_SIZE, voltaje, corriente):

        while GLOBAL_STATE["running"]:

            x = safe_read_axis(DATAX0)
            y = safe_read_axis(DATAX0 + 2)
            z = safe_read_axis(DATAX0 + 4)
            temp = safe_read_temp()

            mag = float(np.sqrt(x*x + y*y + z*z))

            with LOCK:
                GLOBAL_STATE["mag_buffer"].append(mag)
                GLOBAL_STATE["win_buffer"].append(mag)

                if len(GLOBAL_STATE["win_buffer"]) >= WINDOW_SIZE:

                    arr = np.array(GLOBAL_STATE["win_buffer"][-WINDOW_SIZE:])
                    rms = float(np.sqrt(np.mean(arr**2)))

                    fft_vals = np.fft.fft(arr)
                    freqs = np.fft.fftfreq(len(arr), d=1.0/FS)

                    mask = freqs > 0
                    freqs = freqs[mask]
                    fft_vals = np.abs(fft_vals[mask])

                    freq_dom = float(freqs[np.argmax(fft_vals)]) if len(fft_vals) > 0 else 0.0

                    feature_dict = {
                        "rms": rms,
                        "freq": freq_dom,
                        "temp": float(temp),
                        "voltaje": voltaje,
                        "corriente": corriente,
                        "fs": FS,
                        "window": WINDOW_SIZE
                    }

                    try:
                        df_features = pd.DataFrame([feature_dict])
                        df_features = df_features[feature_order]
                        pred = model.predict(df_features)[0]
                    except Exception as e:
                        pred = f"Error: {e}"

                    GLOBAL_STATE["last_prediction"] = pred
                    GLOBAL_STATE["pred_history"].append(pred)
                    GLOBAL_STATE["pred_time"].append(time.time())
                    GLOBAL_STATE["pred_binary"].append(1 if pred in FAIL_STATES else 0)

            time.sleep(1.0 / FS)

    # ========================
    # CONTROLES
    # ========================
    col1, col2 = st.columns(2)

    if col1.button("▶️ Iniciar"):
        if model is None:
            st.error("Carga un modelo primero")
        elif st.session_state.thread is None:
            GLOBAL_STATE["running"] = True
            st.session_state.thread = threading.Thread(
                target=worker_loop,
                args=(model, feature_order, FS, WINDOW_SIZE, voltaje, corriente),
                daemon=True
            )
            st.session_state.thread.start()

    if col2.button("⏹️ Detener"):
        GLOBAL_STATE["running"] = False
        st.session_state.thread = None

    # ========================
    # UI PROFESIONAL
    # ========================
    st.header("📊 Estado del Sistema")

    with LOCK:
        signal = GLOBAL_STATE["mag_buffer"][-200:]
        history = GLOBAL_STATE["pred_history"]
        binary = GLOBAL_STATE["pred_binary"]
        pred = GLOBAL_STATE["last_prediction"]

    # Métricas
    colA, colB, colC = st.columns(3)
    colA.metric("Estado actual", pred)
    colB.metric("Muestras señal", len(signal))
    colC.metric("Predicciones", len(history))

    # Señal
    st.subheader("📈 Señal de vibración")
    if len(signal) > 10:
        df_signal = pd.DataFrame({"Muestra": range(len(signal)), "Aceleración (g)": signal})
        st.line_chart(df_signal.set_index("Muestra"))
    else:
        st.warning("Esperando señal...")

    # Clasificación
    st.subheader("🚨 Clasificación (0=Normal, 1=Falla)")
    if len(binary) > 5:
        df_bin = pd.DataFrame({"Tiempo": range(len(binary)), "Estado": binary})
        st.line_chart(df_bin.set_index("Tiempo"))
    else:
        st.warning("Esperando predicciones...")

    # Distribución
    st.subheader("📊 Distribución de estados")
    if len(history) > 0:
        st.bar_chart(pd.Series(history).value_counts())

    # Tabla
    st.subheader("📋 Historial reciente")
    df_hist = pd.DataFrame({
        "Índice": range(len(history)),
        "Estado": history,
        "Falla (1=Sí)": binary
    })
    st.dataframe(df_hist.tail(20))

    # ========================
    # REFRESH
    # ========================
    time.sleep(0.5)
    st.rerun()
