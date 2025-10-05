import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from io import BytesIO

st.set_page_config(page_title="Advanced PID Vessel Simulator", layout="centered")
st.title("⚙️ Advanced PID Vessel Simulator (Live & Interactive)")

st.markdown("""
Simulate vessel-like processes with live PID control.  
Supports **5 process types**, **load disturbances**, **noise**, **step/ramp/setpoint profiles**, and **CSV export**.  
Use the sidebar to tune PID and process parameters in real-time.
""")

# --- Persistent session state ---
if "running" not in st.session_state: st.session_state.running = False
if "paused" not in st.session_state: st.session_state.paused = False
if "data" not in st.session_state: st.session_state.data = {"t": [], "pv": [], "sp": [], "u": []}
if "integral" not in st.session_state: st.session_state.integral = 0.0
if "prev_error" not in st.session_state: st.session_state.prev_error = 0.0
if "pv" not in st.session_state: st.session_state.pv = 0.0
if "y_dot" not in st.session_state: st.session_state.y_dot = 0.0
if "control_signal" not in st.session_state: st.session_state.control_signal = deque([0.0]*5)
if "start_time" not in st.session_state: st.session_state.start_time = None

# --- Sidebar Controls ---
st.sidebar.header("🎛 PID Controller")
Kp = st.sidebar.slider("Kp", 0.0, 200.0, 10.0, 0.5)
Ki = st.sidebar.slider("Ki", 0.0, 100.0, 1.0, 0.5)
Kd = st.sidebar.slider("Kd", 0.0, 100.0, 1.0, 0.5)
setpoint_mode = st.sidebar.selectbox("Setpoint Profile", ["Constant", "Step", "Ramp", "Sinusoidal"])
setpoint_base = st.sidebar.number_input("Base Setpoint", value=500.0, step=10.0)
setpoint_amp = st.sidebar.number_input("Amplitude (for step/ramp/sinusoid)", value=200.0, step=10.0)
setpoint_time = st.sidebar.number_input("Ramp / Step duration (s)", value=20.0, step=1.0)

st.sidebar.header("🧠 Process Parameters")
process_type = st.sidebar.selectbox("Process Type", ["1st-order lag", 
                                                     "2nd-order underdamped", 
                                                     "2nd-order overdamped",
                                                     "1st-order with dead time",
                                                     "Custom process"])
K_process = st.sidebar.slider("Process Gain (K)", 0.1, 10.0, 1.0, 0.1)
tau = st.sidebar.slider("Time Constant (τ)", 0.1, 10.0, 2.0, 0.1)
zeta = st.sidebar.slider("Damping (ζ) - 2nd order only", 0.0, 2.0, 0.7, 0.05)
dead_time = st.sidebar.slider("Dead Time (s)", 0.0, 5.0, 0.5, 0.1)
noise_std = st.sidebar.slider("Measurement Noise Std Dev", 0.0, 20.0, 0.0, 0.1)
vessel_max = st.sidebar.number_input("Vessel Max Capacity", value=1000.0, step=10.0)
vessel_min = st.sidebar.number_input("Vessel Min Capacity", value=0.0, step=10.0)

dt = 0.05
chart_placeholder = st.empty()

# --- Control buttons ---
col1, col2, col3, col4 = st.columns(4)
start_btn = col1.button("▶ Start / Resume")
pause_btn = col2.button("⏸ Pause")
reset_btn = col3.button("🔁 Reset")
download_btn = col4.button("💾 Download CSV")

# --- Button logic ---
if start_btn:
    st.session_state.running = True
    st.session_state.paused = False
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()
if pause_btn:
    st.session_state.paused = True
    st.session_state.running = False
if reset_btn:
    st.session_state.running = False
    st.session_state.paused = False
    st.session_state.start_time = None
    st.session_state.data = {"t": [], "pv": [], "sp": [], "u": []}
    st.session_state.integral = 0.0
    st.session_state.prev_error = 0.0
    st.session_state.pv = 0.0
    st.session_state.y_dot = 0.0
    st.session_state.control_signal = deque([0.0]*max(1, int(dead_time/dt)))
    st.toast("Simulation reset ✅")
if download_btn:
    csv = "time_s,pv,setpoint,control\n"
    for t,pv_val,sp_val,u_val in zip(st.session_state.data["t"], st.session_state.data["pv"], st.session_state.data["sp"], st.session_state.data["u"]):
        csv += f"{t:.2f},{pv_val:.3f},{sp_val:.3f},{u_val:.3f}\n"
    b = BytesIO()
    b.write(csv.encode())
    b.seek(0)
    st.download_button("Download CSV", b, file_name="pid_simulation.csv", mime="text/csv")

# --- Simulation loop ---
if st.session_state.running and not st.session_state.paused:
    while st.session_state.running and not st.session_state.paused:
        # --- Time ---
        t = time.time() - st.session_state.start_time

        # --- Determine setpoint based on profile ---
        if setpoint_mode == "Constant":
            sp = setpoint_base
        elif setpoint_mode == "Step":
            sp = setpoint_base + (setpoint_amp if t >= setpoint_time else 0.0)
        elif setpoint_mode == "Ramp":
            sp = setpoint_base + min(t/setpoint_time, 1.0) * setpoint_amp
        elif setpoint_mode == "Sinusoidal":
            sp = setpoint_base + setpoint_amp * np.sin(2*np.pi*t/setpoint_time)
        sp = max(vessel_min, min(vessel_max, sp))

        # --- PID calculations ---
        error = sp - st.session_state.pv
        st.session_state.integral += error * dt
        derivative = (error - st.session_state.prev_error) / dt
        st.session_state.prev_error = error
        u = Kp*error + Ki*st.session_state.integral + Kd*derivative

        # --- Dead time ---
        st.session_state.control_signal.append(u)
        delayed_u = st.session_state.control_signal.popleft()

        # --- Process model updates ---
        if process_type == "1st-order lag" or process_type == "1st-order with dead time" or process_type == "Custom process":
            st.session_state.pv += dt * (-(st.session_state.pv) + K_process*delayed_u)/tau
        elif process_type.startswith("2nd-order"):
            wn = 4.0 / max(0.1, tau)
            st.session_state.y_dot += dt * (K_process*wn**2*delayed_u - 2*zeta*wn*st.session_state.y_dot - (wn**2)*st.session_state.pv)
            st.session_state.pv += dt * st.session_state.y_dot

        # --- Add noise ---
        st.session_state.pv += np.random.normal(0.0, noise_std) if noise_std > 0 else 0.0

        # --- Apply vessel limits ---
        st.session_state.pv = max(vessel_min, min(vessel_max, st.session_state.pv))

        # --- Store data ---
        st.session_state.data["t"].append(t)
        st.session_state.data["pv"].append(st.session_state.pv)
        st.session_state.data["sp"].append(sp)
        st.session_state.data["u"].append(u)

        # --- Plot ---
        fig, ax = plt.subplots()
        ax.plot(st.session_state.data["t"], st.session_state.data["pv"], label="PV", linewidth=2)
        ax.plot(st.session_state.data["t"], st.session_state.data["sp"], "r--", label="Setpoint")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Vessel Output")
        ax.legend()
        ax.grid(True)
        chart_placeholder.pyplot(fig)

        time.sleep(dt)

st.write("---")
st.caption("Adjust PID, process, and vessel parameters live. Use pause/resume/reset. Download CSV anytime.")
