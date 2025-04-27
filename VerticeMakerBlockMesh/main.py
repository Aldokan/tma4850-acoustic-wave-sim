# File: main.py

import sys
from vertice_manager import VerticeManager
from plate import Plate
from plot_geometry import plot_2d_lines, print_blockMeshDict

def safe_input(prompt):
    sys.stderr.write(prompt)
    return sys.stdin.readline().strip()

def main():
    vm = VerticeManager(8.0, 6.0)
    
    # Number of plates
    plate_count_str = safe_input("How many plates? ")
    try:
        i = int(plate_count_str)
    except ValueError:
        i = 0

    for idx in range(i):
        raw_line = safe_input(f"Plate #{idx+1} [side [bottom,top,left,right], start, end]: ")
        parts = raw_line.split()
        if len(parts) != 3:
            sys.stderr.write(f"Skipping invalid input: {raw_line}\n")
            continue
        
        side, start_str, end_str = parts[0], parts[1], parts[2]
        p = Plate(side, start_str, end_str)
        vm.add_plate(p)
    
    verticals   = vm.get_vertical_lines()
    horizontals = vm.get_horizontal_lines()
    points_2d   = vm.compute_points_2d()
    blocks      = vm.compute_hex(points_2d)
    points_3d   = vm.convert_to_3d(height=0.01)

    # Print blockMeshDict to stdout
    print_blockMeshDict(vm, points_3d, blocks, points_2d)

    # If you want to visually debug:
    show_plot = False
    if show_plot:
        plot_2d_lines(vm.L, vm.H, verticals, horizontals, points_2d)

if __name__ == "__main__":
    main()

