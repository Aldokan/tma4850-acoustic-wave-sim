#!/usr/bin/env python3
"""
integrated_pressure.py

This module provides the candidate extraction function get_plate_candidates
for an automated plate placement optimization in a 2D room acoustics simulation.

It reads the pressure data from VTP files (assumed to be named 'reflectingWall.vtp')
located in the "boundary" subdirectories of candidate directories in the specified vtk_root folder.
The directories must have a name starting with a given folder_prefix.

Each VTP file is processed via process_vtp_file which:
  - Loads the cell-center pressure data from the file,
  - Classifies each cell onto a wall boundary (left, right, top, or bottom) using a
    classification routine (here implemented in classify_wall_and_bin_coord),
  - And returns a DataFrame with columns 'wall', 'binCoord', and 'pressure'.

The room is assumed to be 8 m wide (for top and bottom walls) and 6 m high (for left and right walls).
The walls are discretized into 1‑meter candidate segments. For each candidate, the integrated pressure
(i.e. the sum of pressure from all cells whose bin coordinate falls within that 1‑meter bin)
is computed. The candidate segments are then sorted (highest integrated pressure first)
and returned as a list of candidate specification strings of the form:
    "<wall> <start> <end>"

If plot_distribution is True, a bar plot displaying the integrated pressures over the candidate bins is generated.
"""

import os
import pandas as pd
import pyvista as pv

# --- Classification Function ---
def classify_wall_and_bin_coord(x, y, tol=1e-3, domain_width=8.0, domain_height=6.0):
    """
    Classify a cell center at (x, y) into a wall boundary.
    
    Returns:
      (wall, bin_coord)
      
      wall: one of 'left', 'right', 'top', or 'bottom' (or None if not near a wall).
      bin_coord: the coordinate along the wall to be used for binning.
      
    For vertical walls ('left' and 'right'), bin_coord is the y coordinate.
    For horizontal walls ('top' and 'bottom'), bin_coord is the x coordinate.
    """
    if abs(x) < tol:
        return ("left", y)
    elif abs(x - domain_width) < tol:
        return ("right", y)
    elif abs(y) < tol:
        return ("bottom", x)
    elif abs(y - domain_height) < tol:
        return ("top", x)
    return (None, None)

# --- Process VTP File ---
def process_vtp_file(vtp_file, tol=1e-3, domain_width=8.0, domain_height=6.0):
    """
    Load a reflectingWall.vtp file, compute cell centers, classify each cell
    to determine on which wall it is located, and return a DataFrame with columns:
      'wall'     : Wall label
      'binCoord' : Coordinate used for binning
      'pressure' : Pressure value
    """
    mesh = pv.read(vtp_file)
    cell_centers = mesh.cell_centers()
    coords = cell_centers.points  # shape (N, 3)
    
    if "pa" in cell_centers.point_data:
        pressures = cell_centers.point_data["pa"]
    else:
        raise KeyError("No 'pa' field in cell_centers.point_data.")
    
    wall_labels = []
    bin_values = []
    for (x, y, z) in coords:
        label, bin_val = classify_wall_and_bin_coord(x, y, tol=tol,
                                                       domain_width=domain_width,
                                                       domain_height=domain_height)
        wall_labels.append(label)
        bin_values.append(bin_val)
        
    df = pd.DataFrame({
        "wall": wall_labels,
        "binCoord": bin_values,
        "pressure": pressures
    })
    # Drop rows where 'wall' is None (i.e., cells not on a recognized boundary).
    df = df.dropna(subset=["wall"])
    return df

# --- Main Candidate Extraction Function ---
def get_plate_candidates(vtk_root, folder_prefix, plate_length, plot_distribution=False):
    """
    Compute integrated pressure on discretized wall segments based on pressure data
    from VTP files.

    Parameters:
      vtk_root (str): Directory where the VTK files are located.
      folder_prefix (str): Prefix for directories containing VTP data.
      plate_length (float): Candidate plate segment length in meters (e.g., 1.0).
      plot_distribution (bool): If True, display a bar plot of the integrated pressures.

    Returns:
      List[str]: A sorted list (highest integrated pressure first) of candidate plate
                 specifications in the form "<wall> <start> <end>".
    """
    # Find directories in vtk_root that start with the folder_prefix.
    candidate_dirs = [os.path.join(vtk_root, d) for d in os.listdir(vtk_root)
                      if d.startswith(folder_prefix) and os.path.isdir(os.path.join(vtk_root, d))]
    
    if not candidate_dirs:
        raise ValueError(f"No directories found in {vtk_root} with prefix '{folder_prefix}'.")
    
    df_list = []
    for cand_dir in candidate_dirs:
        # Look for the VTP file inside the "boundary" subdirectory.
        vtp_file = os.path.join(cand_dir, "boundary", "reflectingWall.vtp")
        if os.path.exists(vtp_file):
            try:
                df = process_vtp_file(vtp_file, tol=1e-3, domain_width=8.0, domain_height=6.0)
                df_list.append(df)
            except Exception as e:
                print(f"Error processing file {vtp_file}: {e}")
        else:
            print(f"File '{vtp_file}' not found in directory {cand_dir}.")
    
    if not df_list:
        raise ValueError("No valid VTP files were processed.")
    
    full_df = pd.concat(df_list, ignore_index=True)
    candidate_pressures = {}
    
    # Process each wall separately.
    walls = full_df['wall'].unique()
    for wall in walls:
        if wall in ['left', 'right']:
            # Use y coordinate for vertical walls.
            min_coord, max_coord = 0.0, 6.0
        elif wall in ['top', 'bottom']:
            # Use x coordinate for horizontal walls.
            min_coord, max_coord = 0.0, 8.0
        else:
            continue

        bins = []
        current = min_coord
        eps = 1e-6  # small epsilon for floating-point comparisons
        while current + plate_length <= max_coord + eps:
            bins.append((current, current + plate_length))
            current += plate_length

        for bin_start, bin_end in bins:
            mask = (
                (full_df['wall'] == wall) &
                (full_df['binCoord'] >= bin_start) &
                (full_df['binCoord'] < bin_end)
            )
            integrated_pressure = full_df.loc[mask, 'pressure'].sum()
            spec = f"{wall} {bin_start} {bin_end}"
            candidate_pressures[spec] = integrated_pressure

    print("Computed integrated pressures for candidate plate segments:")
    for spec, pressure in candidate_pressures.items():
        print(f"  {spec}: {pressure:.2f}")

    # Optionally, plot the pressure distribution using matplotlib.
    if plot_distribution:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is not installed; skipping plot.")
        else:
            # Sort the candidates to match the order below.
            sorted_keys = sorted(candidate_pressures, key=lambda k: candidate_pressures[k], reverse=True)
            pressures = [candidate_pressures[k] for k in sorted_keys]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(sorted_keys, pressures)
            ax.set_xlabel('Candidate Bin (Wall start end)')
            ax.set_ylabel('Integrated Pressure')
            ax.set_title('Pressure Distribution Over Candidate Bins')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.show()

    # Sort candidates by integrated pressure (highest first).
    sorted_candidates = sorted(candidate_pressures.items(), key=lambda item: item[1], reverse=True)
    candidate_specs = [spec for spec, pressure in sorted_candidates]
    
    return candidate_specs

if __name__ == "__main__":
    # For testing this module independently.
    vtk_root = "VTK"
    folder_prefix = "AFAWT_blockMesh_"
    plate_length = 1.0
    try:
        # Set plot_distribution=True to see the bar plot.
        candidates = get_plate_candidates(vtk_root, folder_prefix, plate_length, plot_distribution=True)
        print("\nCandidate plate specifications (sorted by integrated pressure descending):")
        for cand in candidates:
            print(f"  {cand}")
    except Exception as e:
        print("Error:", e)
