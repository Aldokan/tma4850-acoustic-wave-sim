from plate import Plate
from collections import defaultdict
from typing import List, Tuple
import math

# -------------------------
# VerticeManager definition
# -------------------------
class VerticeManager:
    def __init__(self, L: float, H: float):
        self.L = L
        self.H = H
        self.plates = []
        # Always store domain boundary lines
        self.vertical_lines   = set([0.0, self.L])
        self.horizontal_lines = set([0.0, self.H])

    def add_plate(self, plate: Plate):
        """
        Add a plate and update the splitting lines.
        """
        self.plates.append(plate)
        if plate.side in ['bottom', 'top']:
            xA, xB = plate.start, plate.end
            # Round again to ensure consistent lines
            xA = round(xA, 2)
            xB = round(xB, 2)
            self.vertical_lines.add(xA)
            self.vertical_lines.add(xB)
        else:
            yA, yB = plate.start, plate.end
            yA = round(yA, 2)
            yB = round(yB, 2)
            self.horizontal_lines.add(yA)
            self.horizontal_lines.add(yB)

    def get_vertical_lines(self):
        return sorted(self.vertical_lines)

    def get_horizontal_lines(self):
        return sorted(self.horizontal_lines)

    def compute_points_2d(self) -> List[Tuple[float, float]]:
        """
        Build the set of 2D points from:
          1) Domain corners,
          2) Plate endpoints (and their opposites),
          3) Intersections of all vertical and horizontal lines.
        Returns a sorted list (sorted by y then x).
        """
        points_set = set()

        # Domain corners
        points_set.update([
            (0.0, 0.0),
            (self.L, 0.0),
            (self.L, self.H),
            (0.0, self.H),
        ])

        # Include plate endpoints and their opposites
        for plate in self.plates:
            if plate.side in ['bottom', 'top']:
                xA, xB = sorted([plate.start, plate.end])
                if plate.side == 'bottom':
                    y, y_opp = 0.0, self.H
                else:  # top
                    y, y_opp = self.H, 0.0
                points_set.update([
                    (xA, y), (xB, y),
                    (xA, y_opp), (xB, y_opp),
                ])
            else:  # left/right
                yA, yB = sorted([plate.start, plate.end])
                if plate.side == 'left':
                    x, x_opp = 0.0, self.L
                else:  # right
                    x, x_opp = self.L, 0.0
                points_set.update([
                    (x, yA), (x, yB),
                    (x_opp, yA), (x_opp, yB),
                ])

        # Add intersections of vertical/horizontal lines
        for vx in self.vertical_lines:
            for vy in self.horizontal_lines:
                if 0.0 <= vx <= self.L and 0.0 <= vy <= self.H:
                    points_set.add((vx, vy))

        return sorted(points_set, key=lambda p: (p[1], p[0]))

    def convert_to_3d(self, height: float = 0.01) -> List[Tuple[float, float, float]]:
        """
        Convert each (x,y) point to both a bottom (z=0) and a top (z=height) vertex.
        """
        points_2d = self.compute_points_2d()
        bottom_layer = [(x, y, 0.0) for (x, y) in points_2d]
        top_layer = [(x, y, height) for (x, y) in points_2d]
        return bottom_layer + top_layer


    def compute_hex(self, points_2d: List[Tuple[float, float]]) -> List[Tuple[int, int, int, int, int, int, int, int]]:
        """
        For each rectangular cell defined by the grid points, create a hex block.
        The hex is defined by 8 vertex indices (first 4 for bottom, next 4 for top).
        """
        z_offset = len(points_2d)
        row_map = defaultdict(list)
        for idx, (x, y) in enumerate(points_2d):
            row_map[y].append((x, idx))
        sorted_y_values = sorted(row_map.keys())
        # Ensure each row is sorted in increasing x
        for y in sorted_y_values:
            row_map[y] = sorted(row_map[y], key=lambda t: t[0])
        blocks = []
        for r in range(len(sorted_y_values) - 1):
            row_lower = row_map[sorted_y_values[r]]
            row_upper = row_map[sorted_y_values[r + 1]]
            for c in range(len(row_lower) - 1):
                _, v0 = row_lower[c]
                _, v1 = row_lower[c + 1]
                _, v2 = row_upper[c + 1]
                _, v3 = row_upper[c]
                v4 = v0 + z_offset
                v5 = v1 + z_offset
                v6 = v2 + z_offset
                v7 = v3 + z_offset
                blocks.append((v0, v1, v2, v3, v4, v5, v6, v7))
        return blocks
