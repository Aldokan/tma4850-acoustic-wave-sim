#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, sosfilt

# File and constants
data_file = "mic_response_logMiddle1mCornerGauss01.txt"
ambient_pressure = 101325.0  # Pa
c = 343.0                    # Speed of sound in m/s

# Ensure the images directory exists for saving plots
os.makedirs("images", exist_ok=True)

# =============================================================================
# 1) Load Mic Data and Plot Impulse Response
# =============================================================================
try:
    data = np.genfromtxt(data_file, dtype=np.complex128, comments='#')
except Exception as e:
    print("Error reading file:", e)
    raise

# Extract time and pressure data
time_axis = np.real(data[:, 0])
pressure_complex = data[:, 1]

# Use the real part for time-domain analysis
pressure_time = np.real(pressure_complex)

# Plot the impulse response (time-domain)
plt.figure()
plt.plot(time_axis, pressure_time, label="Mic Pressure (Real Part)")
plt.xlabel("Time [s]")
plt.ylabel("Pressure at Microphone")
plt.title("Impulse Response")
plt.legend()
plt.grid(True)
plt.savefig("images/impulse_response.png")
plt.show()

# =============================================================================
# 2) Frequency Response (Magnitude Spectrum)
# =============================================================================
# Perform FFT using the full complex signal to preserve phase information
freq_response = np.fft.fft(pressure_complex)
dt = time_axis[1] - time_axis[0]
freqs = np.fft.fftfreq(len(pressure_complex), d=dt)

plt.figure()
positive = freqs >= 0  # Consider only positive frequencies
plt.plot(freqs[positive], np.abs(freq_response)[positive], label="Magnitude")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("Frequency Response")
plt.xlim(0, 10000)
plt.ylim(0, 1e5)
plt.legend()
plt.grid(True)
plt.savefig("images/frequency_response.png")
plt.show()

# =============================================================================
# 3) Compute Acoustic Metrics (Full-Band Decay Curve, RT60 and C50)
# =============================================================================
# Subtract the ambient pressure to obtain pressure fluctuations.
p_fluct = pressure_time - ambient_pressure

# Use the squared pressure as a proxy for energy.
p_squared = p_fluct**2

# Compute the Energy Decay Curve (EDC) by performing a cumulative sum in reverse.
edc = np.cumsum(p_squared[::-1])[::-1]
edc_norm = edc / (edc[0] + 1e-12)  # Normalize to avoid division by zero
edc_db = 10 * np.log10(edc_norm + 1e-12)  # Convert to decibels (dB)

# --- RT60 Calculation using the EDT method ---
# Select the decay range between 0 dB and -10 dB.
indices_EDT = np.where((edc_db <= 0) & (edc_db >= -10))[0]
if len(indices_EDT) > 1:
    # Extract time and corresponding dB values for this range.
    t_reg = time_axis[indices_EDT]
    db_reg = edc_db[indices_EDT]
    # Perform a linear regression to determine the decay slope (in dB/s).
    slope, intercept = np.polyfit(t_reg, db_reg, 1)
    # Calculate the time for a 10 dB drop.
    EDT_val = 10 / abs(slope)
    # Approximate RT60 by scaling the EDT.
    RT60 = EDT_val * 6
else:
    RT60 = None

# --- C50 Calculation (Speech Clarity) ---
# Define a cutoff time of 50 ms.
t_cutoff = 0.05  # seconds
indices_early = np.where(time_axis <= t_cutoff)[0]
indices_late = np.where(time_axis > t_cutoff)[0]

energy_early = np.trapezoid(p_squared[indices_early], time_axis[indices_early])
energy_late = np.trapezoid(p_squared[indices_late], time_axis[indices_late])
C50 = 10 * np.log10(energy_early / (energy_late + 1e-12))

# --- Plot the Energy Decay Curve and EDT Regression ---
plt.figure(figsize=(10, 6))
plt.plot(time_axis, edc_db, label="EDC (dB)")
if len(indices_EDT) > 1:
    plt.plot(t_reg, db_reg, 'o', color='cyan', label="EDT Range")
    t_line = np.linspace(t_reg[0], t_reg[-1], 100)
    db_line = slope * t_line + intercept
    plt.plot(t_line, db_line, 'r--', label="Linear Regression")
else:
    plt.text(0.1, -5, "Insufficient data for EDT regression", color="red")
plt.xlabel("Time [s]")
plt.ylabel("EDC (dB)")
plt.title("Energy Decay Curve and EDT Regression")
plt.legend()
plt.grid(True)
plt.xlim(0, 0.25)
plt.ylim(min(edc_db) - 5, 1)
plt.savefig("images/decay_curve.png")
plt.show()