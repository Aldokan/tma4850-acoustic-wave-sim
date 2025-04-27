#!/usr/bin/env python3
import os
import subprocess
import time
import sys

from post_process import compute_acoustic_metrics
from integrated_pressure_new import get_plate_candidates

# ------------- PARAMETERS -------------
MAX_ITER = 10

# Performance targets:
TARGET_RT_LOW = 0.3   # seconds
TARGET_RT_HIGH = 0.4  # seconds
TARGET_C50 = 0.0      # dB

# Candidate extraction parameters:
ROOT_DIR = "VTK"                # Directory with VTK files
FOLDER_PREFIX = "AFAWT_blockMesh_"
SEGMENT_WIDTH = 1.0             # Candidate window length

# Log file for plate configurations.
LOG_FILE = "log.txt"  
# New iteration log file to record each iteration details.
ITERATION_LOG_FILE = "iteration_log.txt"

plates = []           # Global list of plate specifications

# ------------- SYMMETRIC PLATE FUNCTION -------------
def get_symmetric_plate_spec(spec):
    """
    Returns the symmetric spec of a candidate.
    
    For example, a candidate "left 2.0 3.0" will be converted to "right 2.0 3.0",
    and "top 4.0 5.0" becomes "bottom 4.0 5.0". If there is no symmetric counterpart,
    returns the original spec.
    """
    parts = spec.split()
    if len(parts) != 3:
        return spec
    wall, start, end = parts
    wall = wall.lower()
    sym_map = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
    if wall in sym_map:
        return f"{sym_map[wall]} {start} {end}"
    return spec

# ------------- SIMULATION WORKFLOW FUNCTIONS -------------

def run_simulation():
    """
    Run the simulation steps (blockMesh, acousticFoam, setManufactured, etc.).
    """
    cmds = [
        ["blockMesh"],
        ["acousticFoam", "-postProcess", "-func", "writeCellCentres"],
        ["python3", "setManufactured.py"],
        ["acousticFoam"],
        ["foamToVTK"]
    ]
    for cmd in cmds:
        print(f"\nRunning command: {' '.join(cmd)}")
        ret = subprocess.run(cmd, cwd=os.getcwd())
        if ret.returncode != 0:
            sys.exit(f"Command {' '.join(cmd)} failed. Exiting.")
    print("Simulation finished.")

def update_allrun_input(plates_list):
    """
    Build the input string for the geometry script from the plate specifications,
    then run the geometry script to update blockMeshDict and rebuild the mesh.
    """
    clean_plates = [p.strip() for p in plates_list if p.strip()]
    input_lines = [str(len(clean_plates))] + clean_plates
    input_str = "\n".join(input_lines)
    print("Updating geometry with these plates:")
    print(input_str)
    
    GEOM_SCRIPT = os.path.join("VerticeMakerBlockMesh", "main.py")
    out_file = os.path.join(os.getcwd(), "system", "blockMeshDict")
    with open(out_file, "w") as f_out:
        ret = subprocess.run(["python3", GEOM_SCRIPT],
                             input=input_str.encode(),
                             stdout=f_out,
                             cwd=os.getcwd())
    if ret.returncode != 0:
        sys.exit("Geometry update failed!")
    print("Geometry updated (blockMeshDict generated).")
    
    # Clean up and reset initial conditions.
    subprocess.run(["rm", "-r", "0"], cwd=os.getcwd())
    subprocess.run(["foamListTimes", "-rm"], cwd=os.getcwd())
    subprocess.run(["cp", "-r", "0.orig", "0"], cwd=os.getcwd())
    ret = subprocess.run(["blockMesh"], cwd=os.getcwd())
    if ret.returncode != 0:
        sys.exit("blockMesh failed after geometry update!")
    print("blockMesh executed successfully.")

def clean_vtk_folder():
    """
    Remove the VTK folder.
    """
    vtk_path = os.path.join(os.getcwd(), "VTK")
    if os.path.exists(vtk_path):
        print(f"Deleting VTK folder at: {vtk_path}")
        subprocess.run(["rm", "-rf", "VTK"], cwd=os.getcwd())
    else:
        print("No VTK folder found.")

def run_postprocessing():
    """
    Use compute_acoustic_metrics to obtain RT and C50 from the current case.
    """
    case_dir = os.getcwd()
    RT, C50 = compute_acoustic_metrics(case_dir)
    return RT, C50

# ------------- LOG FILE & PLATE PARSING FUNCTIONS -------------

def load_plate_configurations():
    """
    Load plate configurations from a log file.
    """
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return [line.strip() for line in f if line.strip()]
    return []

def save_plate_configurations(plates_list):
    """
    Save plate configurations to the log file.
    """
    with open(LOG_FILE, "w") as f:
        for p in plates_list:
            f.write(p + "\n")
    print(f"Plate configurations saved to {LOG_FILE}.")

def parse_plate_spec(spec):
    """
    Convert a spec string (e.g., "left 2.5 3.5") into a tuple (wall, start, end).
    """
    parts = spec.split()
    if len(parts) != 3:
        return None
    try:
        wall = parts[0].lower()
        start = float(parts[1])
        end = float(parts[2])
        return (wall, start, end)
    except:
        return None

def log_iteration(iteration, plates, RT, C50):
    """
    Append a log entry for the current iteration to the iteration log file.
    """
    with open(ITERATION_LOG_FILE, "a") as f:
        f.write(f"Iteration {iteration}: Plates: {plates}; RT60: {RT:.3f} s, C50: {C50:.3f} dB\n")

# ------------- MAIN OPTIMIZATION LOOP -------------

def main():
    global plates
    plates = load_plate_configurations()
    print("Loaded plates from log:", plates)
    
    iteration = 0
    while iteration < MAX_ITER:
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")
        
        # Update geometry with current plate configurations.
        update_allrun_input(plates)
        
        # Run simulation steps.
        run_simulation()
        time.sleep(3)
        
        RT, C50 = run_postprocessing()
        print(f"Measured RT: {RT:.3f} s, C50: {C50:.3f} dB")
        
        # Log the current iteration with its plate configuration and performance metrics.
        log_iteration(iteration, plates, RT, C50)
        
        if (TARGET_RT_LOW <= RT <= TARGET_RT_HIGH) and (C50 > TARGET_C50):
            print("Performance targets met! Ending optimization loop.")
            break
        
        print("Performance targets not met. Running candidate extraction for next plate location...")
        
        try:
            new_candidates = get_plate_candidates(
                vtk_root=ROOT_DIR,
                folder_prefix=FOLDER_PREFIX,
                plate_length=SEGMENT_WIDTH,
            )
        except Exception as e:
            sys.exit(f"Error in candidate extraction: {e}")
        
        if not new_candidates:
            sys.exit("No candidate plate specifications were found. Exiting loop.")
        
        print("New candidate(s) identified:", new_candidates)
        
        # Choose the top candidate and generate its symmetric candidate.
        chosen_spec = new_candidates[0]
        symmetric_spec = get_symmetric_plate_spec(chosen_spec)
        
        # Append both candidates if they are not already added.
        if chosen_spec not in plates:
            plates.append(chosen_spec)
        if symmetric_spec not in plates and symmetric_spec != chosen_spec:
            plates.append(symmetric_spec)
            
        print("Updated plates list (including symmetry):", plates)
        
        # Clean VTK and restore initial simulation files.
        subprocess.run(["foamListTimes", "-rm"], cwd=os.getcwd())
        subprocess.run(["cp", "-r", "0.orig", "0"], cwd=os.getcwd())
        clean_vtk_folder()
        
        save_plate_configurations(plates)
        print("Configuration updated. Re-running simulation...\n")
    
    if iteration >= MAX_ITER:
        print("Maximum iterations reached.")
    print("Final plate configuration:", plates)

if __name__ == "__main__":
    main()
