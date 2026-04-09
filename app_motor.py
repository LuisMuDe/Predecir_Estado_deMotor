def run():
    import streamlit as st
    import smbus2
    import numpy as np
    import pandas as pd
    import time

    # ========================
    # CONFIG
    # ========================
    st.set_page_config(layout="wide")

    bus = smbus2.SMBus(1)

    ADXL_ADDR = 0x53
    POWER_CTL = 0x2D
    DATA_FORMAT = 0x31
    DATAX0 = 0x32
    MLX_ADDR = 0x5A

    # ========================
    # FUNCIONES
    # ========================
    def read_axis(addr):
        low = bus.read_byte_data(ADXL_ADDR, addr)
        high = bus.read_byte_data(ADXL_ADDR, addr + 1)
        value = (high << 8) | low
        if value & (1 << 15):
            value -= (1 << 16)
        return value * 0.004

    def read_temp():
        data = bus.read_word_data(MLX_ADDR, 0x07)
        return (data * 0.02) - 273.15

    def model_predict(rms, freq):
        if rms < 0.2:
            return "Normal"
        elif rms < 0.5:
            return "Alerta"
        else:
            return "Falla"

    def nearest_power_of_2(n):
        return int(2**np.round(np.log2(n)))

    # ========================
    # SESSION STATE
    # ========================
    if "running" not in st.session_state:
        st.session_state.running = False

    if "buffer" not in st.session_state:
        st.session_state.buffer = []

    if "data" not in st.session_state:
        st.session_state.data = []

    if "dataset_ml" not in st.session_state:
        st.session_state.dataset_ml = []

    if "voltaje_nominal" not in st.session_state:
        st.session_state.voltaje_nominal = 12.0

    if "corriente_nominal" not in st.session_state:
        st.session_state.corriente_nominal = 1.5

    if "fs" not in st.session_state:
        st.session_state.fs = 10

    if "window_size" not in st.session_state:
        st.session_state.window_size = 64

    # ========================
    # SIDEBAR
    # ========================
    st.sidebar.header("⚙️ Parámetros")

    st.session_state.voltaje_nominal = st.sidebar.number_input(
        "Voltaje nominal (V)", 0.0, 100.0, st.session_state.voltaje_nominal, 0.5
    )

    st.session_state.corriente_nominal = st.sidebar.number_input(
        "Corriente nominal (A)", 0.0, 50.0, st.session_state.corriente_nominal, 0.1
    )

    # ---- FS y WINDOW ----
    st.sidebar.header("📡 Adquisición")

    st.session_state.fs = st.sidebar.number_input(
        "Frecuencia de muestreo FS (Hz)", 1, 200, st.session_state.fs, 1
    )

    st.session_state.window_size = st.sidebar.number_input(
        "Tamaño de ventana", 16, 512, st.session_state.window_size, 1
    )

    # ---- Etiqueta ----
    st.sidebar.header("🏷️ Etiquetado")

    label_options = ["Normal", "Falla"]

    st.session_state.label = st.sidebar.selectbox(
        "Estado del sistema", label_options
    )

    # ---- Guardar dataset ----
    if st.sidebar.button("💾 Guardar dataset ML"):
        df_ml = pd.DataFrame(st.session_state.dataset_ml)
        df_ml.to_csv("dataset_ml_motor.csv", index=False)
        st.sidebar.success("Dataset guardado")

    # ========================
    # VARIABLES DINÁMICAS
    # ========================
    FS = st.session_state.fs
    WINDOW_SIZE = nearest_power_of_2(int(st.session_state.window_size))

    # ========================
    # INIT SENSOR
    # ========================
    bus.write_byte_data(ADXL_ADDR, POWER_CTL, 0x08)
    bus.write_byte_data(ADXL_ADDR, DATA_FORMAT, 0x08)

    # ========================
    # UI PRINCIPAL
    # ========================
    st.title("📊 Monitoreo Motor DC")

    col_btn1, col_btn2, col_btn3 = st.columns(3)

    if col_btn1.button("▶️ Iniciar"):
        st.session_state.running = True

    if col_btn2.button("⏹️ Detener"):
        st.session_state.running = False

    if col_btn3.button("💾 Guardar datos"):
        df = pd.DataFrame(st.session_state.data)
        df.to_csv("datos_motor_streamlit.csv", index=False)
        st.success("Datos guardados")

    # Métricas
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    metric_rms = col1.empty()
    metric_temp = col2.empty()
    metric_estado = col3.empty()
    metric_v = col4.empty()
    metric_i = col5.empty()
    metric_fs = col6.empty()
    metric_win = col7.empty()

    chart_time = st.empty()
    chart_fft = st.empty()

    # ========================
    # LOOP CONTROLADO
    # ========================
    if st.session_state.running:

        x = read_axis(DATAX0)
        y = read_axis(DATAX0 + 2)
        z = read_axis(DATAX0 + 4)
        temp = read_temp()

        mag = np.sqrt(x**2 + y**2 + z**2)
        st.session_state.buffer.append(mag)

        rms = 0
        freq_dom = 0
        estado = "Inicializando"

        if len(st.session_state.buffer) >= WINDOW_SIZE:
            arr = np.array(st.session_state.buffer)

            rms = np.sqrt(np.mean(arr**2))

            fft_vals = np.fft.fft(arr)
            freqs = np.fft.fftfreq(len(arr), d=1/FS)

            mask = freqs > 0
            freqs = freqs[mask]
            fft_vals = np.abs(fft_vals[mask])

            if len(fft_vals) > 0:
                freq_dom = freqs[np.argmax(fft_vals)]

            # DATASET ML
            st.session_state.dataset_ml.append({
                "rms": rms,
                "freq": freq_dom,
                "temp": temp,
                "voltaje": st.session_state.voltaje_nominal,
                "corriente": st.session_state.corriente_nominal,
                "fs": FS,
                "window": WINDOW_SIZE,
                "label": st.session_state.label
            })

            # Auto-save
            if len(st.session_state.dataset_ml) % 20 == 0:
                df_ml = pd.DataFrame(st.session_state.dataset_ml)
                df_ml.to_csv("dataset_ml_motor_autosave.csv", index=False)

            estado = model_predict(rms, freq_dom)

            df_fft = pd.DataFrame({"freq": freqs, "amp": fft_vals})
            chart_fft.line_chart(df_fft.set_index("freq"))

            st.session_state.buffer = []

        # Histórico
        st.session_state.data.append({
            "vibracion": mag,
            "rms": rms,
            "temp": temp,
            "freq": freq_dom,
            "estado": estado
        })

        df = pd.DataFrame(st.session_state.data[-200:])

        # UI
        metric_rms.metric("RMS", f"{rms:.3f} g")
        metric_temp.metric("Temp", f"{temp:.2f} °C")
        metric_estado.metric("Estado", estado)
        metric_v.metric("Voltaje", f"{st.session_state.voltaje_nominal:.1f} V")
        metric_i.metric("Corriente", f"{st.session_state.corriente_nominal:.2f} A")
        metric_fs.metric("FS", f"{FS} Hz")
        metric_win.metric("Window", f"{WINDOW_SIZE}")

        chart_time.line_chart(df[["vibracion", "rms"]])

        time.sleep(1.0 / FS)
        st.rerun()
