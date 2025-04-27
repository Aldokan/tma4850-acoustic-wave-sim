import matplotlib.pyplot as plt
from typing import List, Tuple
from vertice_manager import VerticeManager

def plot_2d_lines(L, H, verticals, horizontals, points_2d):
    plt.figure(figsize=(6, 6))
    for vx in verticals:
        plt.plot([vx, vx], [0, H], 'k-', lw=1)
    for vy in horizontals:
        plt.plot([0, L], [vy, vy], 'k-', lw=1)
    px = [p[0] for p in points_2d]
    py = [p[1] for p in points_2d]
    plt.scatter(px, py, color='red', zorder=3)
    plt.xlim(-0.5, L + 0.5)
    plt.ylim(-0.5, H + 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title("2D Domain with Plate Subdivisions")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def print_openfoam_vertices(points_3d):
    print("vertices")
    print("(")
    for (x, y, z) in points_3d:
        print(f"    ({x} {y} {z})")
    print(");")

def print_openfoam_hex_blocks(blocks, nx=10, ny=10, nz=1):
    print("blocks")
    print("(")
    for block in blocks:
        vert_string = " ".join(str(v) for v in block)
        print(f"    hex ({vert_string}) ({nx} {ny} {nz}) simpleGrading (1 1 1)")
    print(");")


def is_absorbing_for_segment(vm: VerticeManager, side: str, low: float, high: float) -> bool:
    """
    Return True if there is a plate on the given side that fully covers the segment [low, high].
    For bottom/top the segment is in x; for left/right the segment is in y.
    """
    for plate in vm.plates:
        if plate.side == side:
            if plate.start <= low and high <= plate.end:
                return True
    return False

def compute_boundary_faces(vm: VerticeManager, points_2d: List[Tuple[float, float]],
                           blocks: List[Tuple[int, int, int, int, int, int, int, int]]
                           ) -> Tuple[List[Tuple[int, int, int, int]],
                                      List[Tuple[int, int, int, int]],
                                      List[Tuple[int, int, int, int]]]:
    """
    Compute boundary faces for the structured grid.
    - For the four vertical sides (in the xy-plane) we loop over the segments.
      Each segment is classified as absorbing (if a plate covers it) or reflecting.
    - For the extruded (z-direction) faces we simply collect the front and back faces of every hex block.
    Returns three lists of faces: absorbing_faces, reflecting_faces, front_back_faces.
    """
    verticals = vm.get_vertical_lines()
    horizontals = vm.get_horizontal_lines()
    nx = len(verticals)
    ny = len(horizontals)
    offset = nx * ny  # number of vertices in the bottom layer

    absorbing_faces = []
    reflecting_faces = []

    # Left wall (x=0)
    for r in range(ny - 1):
        y_low = horizontals[r]
        y_high = horizontals[r + 1]
        face = (r * nx, (r + 1) * nx, (r + 1) * nx + offset, r * nx + offset)
        if is_absorbing_for_segment(vm, 'left', y_low, y_high):
            absorbing_faces.append(face)
        else:
            reflecting_faces.append(face)

    # Right wall (x = L)
    for r in range(ny - 1):
        y_low = horizontals[r]
        y_high = horizontals[r + 1]
        face = (r * nx + (nx - 1), (r + 1) * nx + (nx - 1),
                (r + 1) * nx + (nx - 1) + offset, r * nx + (nx - 1) + offset)
        if is_absorbing_for_segment(vm, 'right', y_low, y_high):
            absorbing_faces.append(face)
        else:
            reflecting_faces.append(face)

    # Bottom wall (y=0)
    for c in range(nx - 1):
        x_low = verticals[c]
        x_high = verticals[c + 1]
        face = (c, c + 1, c + 1 + offset, c + offset)
        if is_absorbing_for_segment(vm, 'bottom', x_low, x_high):
            absorbing_faces.append(face)
        else:
            reflecting_faces.append(face)

    # Top wall (y = H)
    for c in range(nx - 1):
        x_low = verticals[c]
        x_high = verticals[c + 1]
        # Bottom vertices for the top row:
        v0 = (ny - 1) * nx + c
        v1 = (ny - 1) * nx + c + 1
        # Top vertices corresponding to the bottom ones:
        v0_top = v0 + offset
        v1_top = v1 + offset
        face = (v0, v1, v1_top, v0_top)
        if is_absorbing_for_segment(vm, 'top', x_low, x_high):
            absorbing_faces.append(face)
        else:
            reflecting_faces.append(face)

    # Front and Back faces (extruded faces in z-direction)
    front_back_faces = []
    for block in blocks:
        # For each block, the face on the bottom (z=0) and the face on the top (z=height)
        front_face = (block[0], block[1], block[2], block[3])
        back_face  = (block[4], block[5], block[6], block[7])
        front_back_faces.append(front_face)
        front_back_faces.append(back_face)

    return absorbing_faces, reflecting_faces, front_back_faces

def print_boundary_patches(absorbing_faces: List[Tuple[int, int, int, int]],
                             reflecting_faces: List[Tuple[int, int, int, int]],
                             front_back_faces: List[Tuple[int, int, int, int]]):
    print("boundary")
    print("(")
    print("    absorbingWall")
    print("    {")
    print("        type wall;")
    print("        faces")
    print("        (")
    for face in absorbing_faces:
        face_str = " ".join(str(v) for v in face)
        print(f"            ({face_str})")
    print("        );")
    print("    }")
    print("    reflectingWall")
    print("    {")
    print("        type wall;")
    print("        faces")
    print("        (")
    for face in reflecting_faces:
        face_str = " ".join(str(v) for v in face)
        print(f"            ({face_str})")
    print("        );")
    print("    }")
    print("    frontAndBack")
    print("    {")
    print("        type empty;")
    print("        faces")
    print("        (")
    for face in front_back_faces:
        face_str = " ".join(str(v) for v in face)
        print(f"            ({face_str})")
    print("        );")
    print("    }")
    print(");")

def print_merge_patch_pairs():
    print("mergePatchPairs")
    print("(")
    print(");")

def print_blockMeshDict(vm: VerticeManager, points_3d, blocks, points_2d):
    """
    Print a full blockMeshDict file.
    """
    # Header (foamFile, convertToMeters, etc.)
    print("/*--------------------------------*- C++ -*----------------------------------*\\")
    print("| =========                 |                                                 |")
    print("| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |")
    print("|  \\    /   O peration     | Version: 2406                                   |")
    print("|   \\  /    A nd           | Website:  www.openfoam.com                      |")
    print("|    \\/     M anipulation  |                                                 |")
    print("\\*---------------------------------------------------------------------------*/")
    print("FoamFile")
    print("{")
    print("    version     2.0;")
    print("    format      ascii;")
    print("    class       dictionary;")
    print("    object      blockMeshDict;")
    print("}")
    print("convertToMeters 1.0;")
    print("")
    
    # Vertices and blocks
    print_openfoam_vertices(points_3d)
    print("")
    print_openfoam_hex_blocks(blocks, nx=100, ny=100, nz=1)
    print("")
    
    # Edges (empty in this example)
    print("edges")
    print("(")
    print(");")
    print("")
    
    # Boundaries
    absorbing_faces, reflecting_faces, front_back_faces = compute_boundary_faces(vm, points_2d, blocks)
    print_boundary_patches(absorbing_faces, reflecting_faces, front_back_faces)
    print("")
    
    # Merge patch pairs (empty)
    print_merge_patch_pairs()
