import os
import numpy as np
import matplotlib.pyplot as plt

def compute_acoustic_metrics(case_dir, plot=False):
    """
    Compute acoustic metrics from the probe data stored in the file
    postProcessing/probe/0/pa inside case_dir.

    This function computes:
      - RT60 (reverberation time) using the Energy Decay Curve (EDC) and EDT method.
      - C50 (speech clarity) defined as 10*log10(energy in first 50 ms / energy after 50 ms).
    
    Optionally, if plot=True, a plot is produced showing:
      - The full EDC (in dB)
      - The portion of the curve used for linear regression (EDT range)
      - The fitted linear regression line.

    Returns:
      (RT60, C50)
    """
    # Path to the probe data file.
    filename = os.path.join(case_dir, "postProcessing", "probe", "0", "pa")
    
    # Load data: assume two columns: time (s) and pressure (Pa)
    data = np.genfromtxt(filename, comments='#')
    if data.ndim == 1:  # Only one line exists
        time = np.array([data[0]])
        pressure = np.array([data[1]])
    else:
        time = data[:, 0]
        pressure = data[:, 1]
    
    # Subtract ambient pressure to get fluctuations.
    ambient = 101325.0  # Pa
    p_fluct = pressure - ambient
    
    # Compute squared pressure (energy proxy)
    p_squared = p_fluct**2

    # Compute Energy Decay Curve (EDC)
    edc = np.cumsum(p_squared[::-1])[::-1]
    edc_norm = edc / (edc[0] + 1e-12)  # normalize to avoid division by zero
    edc_db = 10 * np.log10(edc_norm + 1e-12)  # convert to dB

    # --- RT60 Calculation using EDT ---
    # Use the decay from 0 dB to -10 dB.
    indices_EDT = np.where((edc_db <= 0) & (edc_db >= -10))[0]
    if len(indices_EDT) > 1:
        t_reg = time[indices_EDT]
        db_reg = edc_db[indices_EDT]
        # Perform a linear regression to obtain the decay slope in dB/s.
        slope, intercept = np.polyfit(t_reg, db_reg, 1)
        # The time for a 10 dB drop:
        EDT_val = 10 / abs(slope)
        # RT60 is approximated by scaling EDT_val (for instance, by a factor of 6).
        RT60 = EDT_val * 6
    else:
        RT60 = None

    # --- C50 Calculation ---
    # Define a cutoff time of 50 ms.
    t_cutoff = 0.05  # seconds
    indices_early = np.where(time <= t_cutoff)[0]
    indices_late = np.where(time > t_cutoff)[0]
    
    energy_early = np.trapz(p_squared[indices_early], time[indices_early])
    energy_late = np.trapz(p_squared[indices_late], time[indices_late])
    
    C50 = 10 * np.log10(energy_early / (energy_late + 1e-12))
    
    # --- Optional Plot ---
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(time, edc_db, label="EDC (dB)")  # Plot the full decay curve
        if len(indices_EDT) > 1:
            plt.plot(t_reg, db_reg, 'o', label="EDT range")
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
        plt.xlim(0, 0.21)
        plt.ylim(-20, 1)
        plt.show()
    
    return RT60, C50

if __name__ == "__main__":
    CASE_DIR = os.getcwd()
    rt60, c50 = compute_acoustic_metrics(CASE_DIR, plot=True)
    if rt60 is not None:
        print("RT60 (from EDT): {:.3f} s".format(rt60))
    else:
        print("RT60 could not be computed (insufficient data).")
    print("Speech Clarity (C50): {:.3f} dB".format(c50))
