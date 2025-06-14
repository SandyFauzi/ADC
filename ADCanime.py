# ======================================================================
# 1. IMPORTS & LIBRARY SETUP
# ======================================================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numerical_methods as nm
import base64
import io
from PIL import Image

# ======================================================================
# 2. HELPER FUNCTIONS (non-numerical core)
# ======================================================================

def sensor_to_voltage(value, min_val, max_val, v_ref):
    range_val = max_val - min_val
    if range_val == 0:
        return v_ref if value >= max_val else 0
    return ((value - min_val) / range_val) * v_ref

def digital_to_sensor(digital_val, resolution, min_val, max_val):
    max_digital = 2**resolution - 1
    range_val = max_val - min_val
    return ((digital_val / max_digital) * range_val) + min_val

def to_binary_string(n, bits):
    try:
        if np.isnan(n) or np.isinf(n):
            n = 0
        val = int(round(float(n)))
        if val < 0:
            val = 0
    except Exception:
        val = 0
    return format(val, f'0{bits}b')

def calc_rise_time(signal, t_axis):
    if len(signal) < 2:
        return 0.0
    v_min = np.min(signal)
    v_max = np.max(signal)
    if np.isclose(v_max, v_min, atol=1e-5):
        return 0.0
    v_10 = v_min + 0.1 * (v_max - v_min)
    v_90 = v_min + 0.9 * (v_max - v_min)
    idx_10 = None
    idx_90 = None

    for i in range(1, len(signal)):
        if signal[i-1] < v_10 <= signal[i]:
            t1 = t_axis[i-1] + (t_axis[i] - t_axis[i-1]) * (v_10 - signal[i-1]) / (signal[i] - signal[i-1])
            idx_10 = t1
            break
    if idx_10 is None:
        return 0.0

    for i in range(1, len(signal)):
        if t_axis[i-1] >= idx_10:
            if signal[i-1] < v_90 <= signal[i]:
                t2 = t_axis[i-1] + (t_axis[i] - t_axis[i-1]) * (v_90 - signal[i-1]) / (signal[i] - signal[i-1])
                idx_90 = t2
                break
    if idx_90 is None:
        return 0.0

    return idx_90 - idx_10

def max_slew_rate(signal, t_axis, scale_mv=True):
    diffs = np.diff(signal)
    dt = np.diff(t_axis)
    valid = dt > 1e-12
    if not np.any(valid):
        return 0.0
    slew = np.abs(diffs[valid] / dt[valid]) / 1e6
    if scale_mv:
        slew = slew * 1000
    return np.max(slew) if slew.size > 0 else 0.0

def make_signal_animation_gif(x_data, y_data, n_frames, label, title):
    total_len = len(x_data)
    frame_idxs = np.linspace(1, total_len, n_frames, endpoint=True, dtype=int)
    images = []
    for i, idx in enumerate(frame_idxs):
        fig_anim, ax_anim = plt.subplots(figsize=(8,4))
        ax_anim.plot(x_data[:idx], y_data[:idx], color='blue')
        ax_anim.set_xlim(x_data[0], x_data[-1])
        ax_anim.set_ylim(np.min(y_data), np.max(y_data))
        ax_anim.set_title(f"{title}\nFrame {i+1}/{n_frames}")
        ax_anim.set_xlabel("Time (s)")
        ax_anim.set_ylabel(label)
        ax_anim.grid(True)
        buf = io.BytesIO()
        plt.tight_layout()
        fig_anim.savefig(buf, format="png")
        plt.close(fig_anim)
        buf.seek(0)
        images.append(Image.open(buf).convert("RGB"))
    gif_buf = io.BytesIO()
    images[0].save(
        gif_buf, format="GIF", save_all=True, append_images=images[1:],
        duration=int(1000/n_frames), loop=0, disposal=2
    )
    gif_buf.seek(0)
    return gif_buf

# ======================================================================
# 3. MAIN APP FUNCTION (STREAMLIT APP)
# ======================================================================

def main():
    # ========================== HEADER ===============================
    st.set_page_config(page_title="Sensor to ADC Simulator", layout="wide")

    def image_to_base64(file):
        with open(file, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    img_data = image_to_base64("LOGO.png")

    st.markdown(
        f"""
        <style>
            .fixed-logo {{
                position: fixed;
                top: 4.5rem;
                right: 2rem;
                z-index: 100;
                border-radius: 1em;
                padding: 0.5em;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            }}
        </style>
        <div class="fixed-logo">
            <img src="data:image/png;base64,{img_data}" width="140"/>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("🌡️💧🎛️🎤 Sensor Signal Data Simulator")
    st.subheader("Analyzing the Effects of Analog Circuitry on ADC Accuracy")
    st.markdown(
        """
        Kelompok 17 Project Base Learning  
        Komputasi Numerik - Fisika FMIPA Unpad
        """
    )
    # ========================== END HEADER ===============================
    # ------------------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ System Configuration")
        st.subheader("1. Sensor Input")
        sensor_type = st.selectbox("Sensor Type", ["Temperature", "Humidity", "Potentiometer", "Sound (Microphone)"])
        v_ref = st.selectbox("Operating Voltage (Vref)", [5.0, 3.3])

        if sensor_type in ["Temperature", "Humidity", "Potentiometer"]:
            if sensor_type == "Temperature":
                min_sensor, max_sensor, unit = -50.0, 500.0, "°C"
                sensor_value = st.slider(f"Input {sensor_type} ({unit})", min_sensor, max_sensor, 25.0, step=0.01)
            elif sensor_type == "Humidity":
                min_sensor, max_sensor, unit = 0.0, 100.0, "%"
                sensor_value = st.slider(f"Input {sensor_type} ({unit})", min_sensor, max_sensor, 65.0, step=0.01)
            else:  # Potentiometer
                min_sensor, max_sensor, unit = 0.0, 10000.0, "Ω"
                sensor_value = st.slider(f"Input {sensor_type} ({unit})", min_sensor, max_sensor, 5000.0, step=0.01)
        else:  # Sound (Microphone)
            unit = "V"
            frequency = st.slider("Tone Frequency (Hz)", 100, 5000, 440)
            amplitude = st.slider(f"Signal Amplitude ({unit})", 0.1, v_ref / 2, 1.0, step=0.01)

        st.subheader("2. Analog Circuitry")
        res_unit = st.selectbox(
            "Resistor Unit",
            ["Ω", "kΩ", "MΩ"],
            index=1
        )
        res_factor = {"Ω": 1, "kΩ": 1e3, "MΩ": 1e6}[res_unit]
        res_min, res_max, res_default = (0.0, 1000.0, 10.0)
        resistance_val = st.slider(
            f"Filter Resistance ({res_unit})",
            res_min, res_max, res_default, step=0.01
        )
        resistance = resistance_val * res_factor

        cap_unit = st.selectbox(
            "Capacitor Unit",
            ["pF", "nF", "μF", "mF", "F"],
            index=2
        )
        cap_factor = {"pF": 1e-12, "nF": 1e-9, "μF": 1e-6, "mF": 1e-3, "F": 1}[cap_unit]
        cap_min, cap_max, cap_default = (0.01, 100.0, 0.1)
        capacitance_val = st.slider(
            f"Filter Capacitance ({cap_unit})",
            cap_min, cap_max, cap_default, step=0.01
        )
        capacitance = capacitance_val * cap_factor

        gain = st.slider("Op-Amp Gain", 0.1, 10.0, 1.0, step=0.01)
        slew_rate = st.slider("Op-Amp Slew Rate (V/μs)", 0.1, 100.0, 10.0, step=0.01)
        noise_level = st.slider("Noise Level (mV)", 0.0, 10000.0, 10.0, step=0.01)

        st.subheader("3. ADC & Sampling")
        resolution = st.slider("ADC Resolution (bits)", 8, 16, 12)
        sampling_rate = st.slider("Sampling Rate (kSPS)", 1, 1000, 40)
        duration = st.slider("Simulation Duration (ms)", 10, 5000, 100)

        st.subheader("4. Numerical Methods")
        ode_solver_choice = st.selectbox("ODE Solver (RC Filter)", ["Runge-Kutta 4", "Forward Euler"])
        integration_method_choice = st.selectbox("Integration Method (for Error)", ["Trapezoidal", "Simpson's 1/3", "Simpson's 3/8"])
        diff_method_choice = st.selectbox("Differentiation Method (for Slew Rate)", ["Central Difference", "Forward Difference", "Backward Difference"])
        interpolation_method_choice = st.selectbox("Interpolation Method (Reconstruction)", ["Linear", "Cubic"])

        st.subheader("5. Animation (GIF)")
        anim_enable = st.checkbox("Show Animation in separate tab (GIF)?", value=False)
        plot_choice = st.selectbox(
            "Which signal to animate?",
            [
                "Sensor (noisy_signal)",
                "After RC Filter (rc_filtered)",
                "After Op-Amp (opamp_output)",
                "Quantized Voltage (quantized_voltage)",
                "ADC Digital Output (digital_output)"
            ], index=0, disabled=not anim_enable)
        n_frames = st.slider("Number of animation frames", 10, 100, 30, help="More = smoother", disabled=not anim_enable)
        run_simulation = st.button("Run Simulation")
    
    resistance = resistance_val * res_factor
    capacitance = capacitance_val * cap_factor

    # RC TIME CONSTANT VALIDATION
    rc_constant = resistance * capacitance
    if rc_constant < 1e-9 or rc_constant > 10:
        st.error("RC time constant out of reasonable range. Adjust resistance or capacitance value.")
        st.stop()

    # ------------------------------------------------------------------
    # 3.1. SIMULATION CALCULATION
    # ------------------------------------------------------------------
    if not run_simulation:
        st.info("Configure parameters in the sidebar and click 'Run Simulation' to start.")
        return

    t = np.linspace(0, duration / 1000, int(20000 * (duration / 100)), endpoint=False)
    if sensor_type == "Sound (Microphone)":
        ideal_signal = amplitude * np.sin(2 * np.pi * frequency * t) + v_ref / 2
    else:
        ideal_voltage = sensor_to_voltage(sensor_value, min_sensor, max_sensor, v_ref)
        ideal_signal = np.full_like(t, ideal_voltage)

    noisy_signal = ideal_signal + (noise_level / 1000) * np.random.normal(0, 1, len(t))
    rc_constant = resistance * capacitance              

    noisy_signal_func = nm.linear_interpolation(t, noisy_signal)
    def rc_ode_func(t_point, y_point):
        vin_point = float(noisy_signal_func(t_point))
        return (vin_point - y_point) / rc_constant if rc_constant > 1e-9 else 0

    solver_func = nm.runge_kutta_4 if ode_solver_choice == "Runge-Kutta 4" else nm.forward_euler
    t_span = (t[0], t[-1])
    y0 = ideal_signal[0]
    h = t[1] - t[0] if len(t) > 1 else 1
    t_filtered, rc_filtered = solver_func(rc_ode_func, t_span, y0, h)

    opamp_output = gain * rc_filtered
    dt = t_filtered[1] - t_filtered[0] if len(t_filtered) > 1 else 1
    max_dv = slew_rate * dt * 1e6
    for i in range(1, len(opamp_output)):
        dv = opamp_output[i] - opamp_output[i-1]
        if abs(dv) > max_dv:
            opamp_output[i] = opamp_output[i-1] + np.sign(dv) * max_dv
    opamp_output = np.clip(opamp_output, 0, v_ref)

    num_adc_samples = int(sampling_rate * 1000 * (duration / 1000))
    adc_samples_t = np.linspace(0, duration / 1000, num_adc_samples, endpoint=False) if num_adc_samples > 1 else np.array([0])

    opamp_output_func = nm.linear_interpolation(t_filtered, opamp_output)
    adc_input_voltage = opamp_output_func(adc_samples_t)

    max_digital_val = 2 ** resolution - 1
    digital_output = np.round(adc_input_voltage / v_ref * max_digital_val)
    digital_output = np.nan_to_num(digital_output, nan=0, posinf=max_digital_val, neginf=0)
    digital_output = np.clip(digital_output, 0, max_digital_val)
    digital_output = digital_output.astype(int)
    quantized_voltage = digital_output / max_digital_val * v_ref

    if interpolation_method_choice == "Linear":
        interp_func = nm.linear_interpolation(adc_samples_t, quantized_voltage)
    else:
        interp_func = nm.cubic_interpolation(adc_samples_t, quantized_voltage)
    reconstructed_voltage = interp_func(t_filtered)

    if interpolation_method_choice == "Linear":
        f_opamp = nm.linear_interpolation(t_filtered, opamp_output)
    else:
        f_opamp = nm.cubic_interpolation(t_filtered, opamp_output)

    h_slew = t_filtered[1] - t_filtered[0] if len(t_filtered) > 1 else 1
    slew_rate_arr = np.zeros_like(opamp_output)

    for i, tpt in enumerate(t_filtered):
        if diff_method_choice == "Central Difference":
            if i > 0 and i < len(t_filtered) - 1:
                slew_rate_arr[i] = nm.central_difference(f_opamp, tpt, h_slew)
            elif i == 0:
                slew_rate_arr[i] = nm.forward_difference(f_opamp, tpt, h_slew)
            elif i == len(t_filtered) - 1:
                slew_rate_arr[i] = nm.backward_difference(f_opamp, tpt, h_slew)
        elif diff_method_choice == "Forward Difference":
            if i < len(t_filtered) - 1:
                slew_rate_arr[i] = nm.forward_difference(f_opamp, tpt, h_slew)
            else:
                slew_rate_arr[i] = nm.backward_difference(f_opamp, tpt, h_slew)
        else:  # Backward Difference
            if i > 0:
                slew_rate_arr[i] = nm.backward_difference(f_opamp, tpt, h_slew)
            else:
                slew_rate_arr[i] = nm.forward_difference(f_opamp, tpt, h_slew)

    slew_rate_arr = slew_rate_arr * 1e3  # V/s to mV/μs

    st.success("Simulation Complete!")

    # ------------------------------------------------------------------
    # 3.2. OUTPUT TABS
    # ------------------------------------------------------------------
    tabs = ["📊 Signal Visualization", "📈 Performance & Conversion Analysis", "📄 Export Data"]
    if anim_enable:
        tabs.append("🎬 Animation")
    tab1, tab2, tab3, *tab_anim = st.tabs(tabs)

    with tab1:
        st.subheader("Signal's Journey from Sensor to ADC (in Volts)")
        fig, ax = plt.subplots(3, 2, figsize=(14, 15), sharex=True)
        ax = ax.flatten()

        ax[0].plot(t, noisy_signal, 'r', alpha=0.7, label='Original Signal + Noise')
        ax[0].plot(t, ideal_signal, 'b', linestyle='--', label='Ideal Signal')
        ax[0].set_title("1. Signal from Sensor")
        ax[0].legend(loc='upper right', fontsize='small')
        ax[1].plot(t, noisy_signal, 'r', alpha=0.3, label='Before Filter')
        ax[1].plot(t_filtered, rc_filtered, 'g', label=f'After Filter ({ode_solver_choice})')
        ax[1].set_title("2. RC Filter Output")
        ax[1].legend(loc='upper right', fontsize='small')
        ax[2].plot(t_filtered, rc_filtered, 'g', alpha=0.3, label='Before Op-Amp')
        ax[2].plot(t_filtered, opamp_output, 'm', label='After Op-Amp')
        ax[2].set_title("3. Op-Amp Buffer Output")
        ax[2].legend(loc='upper right', fontsize='small')
        ax[3].plot(adc_samples_t, adc_input_voltage, 'c', alpha=0.7, label='Before Quantization')
        ax[3].step(adc_samples_t, quantized_voltage, 'k', where='post', label='After Quantization (Volt)')
        ax[3].plot(t_filtered, reconstructed_voltage, 'orange', linestyle=':', label=f'Reconstructed ({interpolation_method_choice})')
        ax[3].set_title(f"4. Signal at ADC (Voltage)")
        ax[3].legend(loc='upper right', fontsize='small')
        ax[4].step(adc_samples_t, digital_output, 'purple', where='post', label='Raw Digital Value')
        ax[4].set_title(f"5. ADC Encoder Output (Digital Value 0-{int(max_digital_val)})")
        ax[4].set_ylabel("Digital Value (LSB)")
        ax[4].legend(loc='upper right', fontsize='small')
        ax[5].set_title(f"6. ADC Binary Output ({resolution}-bit Logic)")
        if len(digital_output) > 0:
            max_points = 200
            if len(adc_samples_t) > max_points:
                step = max(1, len(adc_samples_t) // max_points)
                indices = np.arange(0, len(adc_samples_t), step)
                adc_samples_t_bin = adc_samples_t[indices]
                digital_output_bin = digital_output[indices]
            else:
                adc_samples_t_bin = adc_samples_t
                digital_output_bin = digital_output
            for i in range(resolution):
                bit_signal = [(int(b[i])) for b in [to_binary_string(val, resolution) for val in digital_output_bin]]
                ax[5].step(adc_samples_t_bin, np.array(bit_signal) + i * 1.5, where='post', label=f'Bit {resolution-1-i}')
            ax[5].set_yticks([(i * 1.5) + 0.5 for i in range(resolution)])
            ax[5].set_yticklabels([f'Bit {resolution-1-i}' for i in range(resolution)])
        for a in ax:
            if a.get_visible():
                a.grid(True)
                a.set_xlabel("Time (s)")
            if a != ax[4] and a != ax[5]:
                a.set_ylabel("Voltage (V)")
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        st.subheader(f"⭐ Final Conversion Result: {sensor_type}")
        if sensor_type != "Sound (Microphone)":
            last_digital_val = digital_output[-1] if len(digital_output) > 0 else 0
            final_value = digital_to_sensor(last_digital_val, resolution, min_sensor, max_sensor)
            actual_value = sensor_value
            col1, col2, col3 = st.columns(3)
            col1.metric(
                f"Actual Input Value",
                f"{actual_value:.2f} {unit}",
                help="The true value entered from the sensor."
            )
            col2.metric(
                f"Measured Output Value",
                f"{final_value:.2f} {unit}",
                help="Value measured after ADC conversion (last sample)."
            )
            col3.metric(
                "Measurement Error",
                f"{final_value - actual_value:+.2f} {unit}",
                help="Difference between input and measured value."
            )
        else:
            output_ac_signal = quantized_voltage - np.mean(quantized_voltage)
            final_value = (np.max(output_ac_signal) - np.min(output_ac_signal)) / 2 if len(output_ac_signal) > 0 else 0
            actual_value = amplitude
            col1, col2, col3 = st.columns(3)
            col1.metric(
                f"Input Amplitude",
                f"{actual_value:.3f} {unit}",
                help="Amplitude set as input for the microphone simulation."
            )
            col2.metric(
                f"Measured Amplitude",
                f"{final_value:.3f} {unit}",
                help="Amplitude measured from the processed output signal."
            )
            col3.metric(
                "Amplitude Error",
                f"{final_value - actual_value:+.3f} {unit}",
                help="Difference between input and measured amplitude."
            )

        st.markdown("---")
        st.subheader("Voltage Signal Quality Evaluation")

        # THRESHOLD VALUES
        rms_ideal_max = 15      # mV
        rise_time_min = 0       # ms
        rise_time_max = 30      # ms
        snr_ideal_min = 45      # dB
        slew_ideal_min = 0      # mV/μs
        slew_ideal_max = 2      # mV/μs

        # Calculate metrics
        ideal_signal_func = nm.linear_interpolation(t, ideal_signal)
        ideal_voltage_sampled = ideal_signal_func(adc_samples_t)
        T_total = adc_samples_t[-1] - adc_samples_t[0] if len(adc_samples_t) > 1 else 1
        rms_error_voltage = 0
        if T_total > 0 and len(adc_samples_t) > 1:
            sq_err_func = nm.linear_interpolation(adc_samples_t, (quantized_voltage - ideal_voltage_sampled)**2)
            n_integral = len(adc_samples_t)
            if integration_method_choice == "Trapezoidal":
                integral_error = nm.trapezoid_rule(sq_err_func, adc_samples_t[0], adc_samples_t[-1], n_integral)
            elif integration_method_choice == "Simpson's 1/3":
                integral_error = nm.simpson_13_rule(sq_err_func, adc_samples_t[0], adc_samples_t[-1], n_integral)
            else:
                integral_error = nm.simpson_38_rule(sq_err_func, adc_samples_t[0], adc_samples_t[-1], n_integral)
            rms_error_voltage = np.sqrt(integral_error / T_total)
        rms_mV = rms_error_voltage*1000

        rise_time_analog = calc_rise_time(opamp_output, t_filtered)
        rise_ms = rise_time_analog*1000

        signal_power = np.mean((ideal_voltage_sampled)**2)
        noise_power = np.mean((quantized_voltage - ideal_voltage_sampled)**2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

        max_slew = max_slew_rate(opamp_output, t_filtered, scale_mv=True)

        # Display each metric with help tooltip
        colA, colB, colC, colD = st.columns(4)
        with colA:
            st.metric(
                label="RMS Error (Voltage)",
                value=f"{rms_mV:.2f} mV",
                help="Root Mean Square Error. Measures average voltage error between ideal and quantized signal (lower is better)."
            )
            st.markdown(
                f"{'✅' if rms_mV <= rms_ideal_max else '❌'} <small>(Ideal ≤ {rms_ideal_max} mV)</small>", unsafe_allow_html=True
            )
        with colB:
            st.metric(
                label="Rise Time (10%-90%)",
                value=f"{rise_ms:.3f} ms",
                help="Time needed for the signal to rise from 10% to 90% of its final value. Lower is better (faster response)."
            )
            st.markdown(
                f"{'✅' if rise_time_min <= rise_ms <= rise_time_max else '❌'} <small>(Ideal {rise_time_min}–{rise_time_max} ms)</small>", unsafe_allow_html=True
            )
        with colC:
            st.metric(
                label="SNR (Signal-to-Noise Ratio)",
                value=f"{snr_db:.2f} dB",
                help="Signal-to-Noise Ratio. Higher SNR means cleaner, less noisy output (higher is better)."
            )
            st.markdown(
                f"{'✅' if snr_db >= snr_ideal_min else '❌'} <small>(Ideal ≥ {snr_ideal_min} dB)</small>", unsafe_allow_html=True
            )
        with colD:
            st.metric(
                label="Slew Rate",
                value=f"{max_slew:.3f} mV/μs",
                help="Maximum rate of voltage change in the signal (how fast the circuit responds to change)."
            )
            st.markdown(
                f"{'✅' if slew_ideal_min <= max_slew <= slew_ideal_max else '❌'} <small>(Ideal {slew_ideal_min}–{slew_ideal_max} mV/μs)</small>", unsafe_allow_html=True
            )

        # ====== BIG, COLORFUL SUMMARY DISPLAY ======
        st.markdown("## 🧾 <b>Quality Evaluation Summary</b>", unsafe_allow_html=True)
        def big_metric(label, value, unit, ok, help_txt):
            color = "#00FF77" if ok else "#FF4444"
            return f'<div style="font-size:2.5rem;font-weight:800;margin-bottom:10px;color:{color};">{label}: {value} <span style="font-size:1.4rem;">{unit}</span><span title="{help_txt}" style="cursor:help;font-size:1.5rem;margin-left:10px;">❓</span></div>'

        rms_ok = rms_mV <= rms_ideal_max
        rise_ok = rise_time_min <= rise_ms <= rise_time_max
        snr_ok = snr_db >= snr_ideal_min
        slew_ok = slew_ideal_min <= max_slew <= slew_ideal_max
        colE, colF = st.columns(2)
        with colE:
            st.markdown(big_metric(
                "RMS Error", f"{rms_mV:.2f}", "mV", rms_ok,
                "Root Mean Square Error. Measures average voltage error between ideal and quantized signal (lower is better)."
            ), unsafe_allow_html=True)
            st.markdown(big_metric(
                "SNR", f"{snr_db:.2f}", "dB", snr_ok,
                "Signal-to-Noise Ratio. Higher SNR means cleaner, less noisy output (higher is better)."
            ), unsafe_allow_html=True)
        with colF:
            st.markdown(big_metric(
                "Rise Time", f"{rise_ms:.3f}", "ms", rise_ok,
                "Time needed for the signal to rise from 10% to 90% of its final value. Lower is better (faster response)."
            ), unsafe_allow_html=True)
            slew_val = f"{max_slew:.3f}" if not np.isnan(max_slew) else "NaN"
            st.markdown(big_metric(
                "Slew Rate", slew_val, "mV/μs", slew_ok,
                "Maximum rate of voltage change in the signal (how fast the circuit responds to change)."
            ), unsafe_allow_html=True)

        # =====================
        st.markdown("---")
        st.subheader("ADC Linearity Analysis (Regression)")
        beta, poly_func = nm.polynomial_regression(adc_input_voltage, digital_output, degree=1)
        std_x = np.std(adc_input_voltage)
        std_y = np.std(digital_output)
        r_squared = np.corrcoef(adc_input_voltage, digital_output)[0,1]**2 if std_x >= 1e-9 and std_y >= 1e-9 else float('nan')
        fig_reg, ax_reg = plt.subplots(figsize=(10, 5))
        ax_reg.scatter(adc_input_voltage, digital_output, alpha=0.3, s=10, label='ADC Sample Data')
        ax_reg.plot(adc_input_voltage, poly_func(adc_input_voltage), 'r-', label=f'Linear Regression Line\nR² = {r_squared:.4f}')
        ax_reg.set_title("ADC Linearity: Input Voltage vs. Digital Output")
        ax_reg.set_xlabel("Input Voltage to ADC (V)")
        ax_reg.set_ylabel(f"Digital Output (0-{int(max_digital_val)})")
        ax_reg.legend(loc='upper left')
        ax_reg.grid(True)
        st.pyplot(fig_reg)


    with tab3:
        st.subheader("Export Simulation Data")
        binary_values = [to_binary_string(val, resolution) for val in digital_output]
        df_opamp = pd.DataFrame({
            'Time (s)': t_filtered,
            'Opamp_Output (V)': opamp_output,
            'Slew_Rate_Analog (mV/μs)': slew_rate_arr
        })
        export_df = pd.DataFrame({
            'Sample_Time (s)': adc_samples_t,
            'ADC_Input_Voltage (V)': adc_input_voltage,
            'Digital_Output': digital_output,
            'Binary_Value': binary_values,
        })
        if sensor_type != "Sound (Microphone)":
            export_df[f'Converted_{sensor_type}_Value ({unit})'] = digital_to_sensor(export_df['Digital_Output'], resolution, min_sensor, max_sensor)
        st.dataframe(export_df)
        st.download_button(
            "Download Detailed Data (CSV)", 
            export_df.to_csv(index=False).encode('utf-8'), 
            f'simulation_data_{sensor_type}.csv'
        )
        st.markdown("Download Analog Output + Slew Rate (Optional):")
        st.download_button(
            "Download Opamp Output & Slew Rate (CSV)",
            df_opamp.to_csv(index=False).encode('utf-8'),
            f'analog_slew_{sensor_type}.csv'
        )

    if anim_enable:
        with tab_anim[0]:
            st.subheader("🎬 Signal Animation (GIF)")
            if plot_choice == "Sensor (noisy_signal)":
                y_data = noisy_signal
                x_data = t
                label = "Sensor Output (V)"
                title = "Sensor Output"
            elif plot_choice == "After RC Filter (rc_filtered)":
                y_data = rc_filtered
                x_data = t_filtered
                label = "After RC Filter (V)"
                title = "After RC Filter"
            elif plot_choice == "After Op-Amp (opamp_output)":
                y_data = opamp_output
                x_data = t_filtered
                label = "Op-Amp Output (V)"
                title = "Op-Amp Output"
            elif plot_choice == "Quantized Voltage (quantized_voltage)":
                y_data = quantized_voltage
                x_data = adc_samples_t
                label = "Quantized Voltage (V)"
                title = "Quantized Voltage"
            else:
                y_data = digital_output
                x_data = adc_samples_t
                label = "ADC Digital Output (LSB)"
                title = "ADC Digital Output"

            gif_buf = make_signal_animation_gif(x_data, y_data, n_frames, label, title)
            st.image(gif_buf, caption=f"{plot_choice} animation")
            st.download_button(
                "Download GIF Animation",
                gif_buf.getvalue(),
                file_name=f"{plot_choice.replace(' ','_').lower()}_animation.gif",
                mime="image/gif"
            )

# ======================================================================
# 4. RUN APP
# ======================================================================

if __name__ == "__main__":
    main()
