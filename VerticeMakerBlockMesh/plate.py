# File: plate.py

class Plate:
    def __init__(self, side: str, start: float, end: float):
        # Convert to lowercase, strip whitespace
        self.side = side.lower().strip()
        
        # Round the start/end to 2 decimal places
        self.start = round(float(start), 2)
        self.end   = round(float(end), 2)
        
        if self.side not in ['bottom', 'top', 'left', 'right']:
            raise ValueError(f"Invalid plate side: {self.side}")
        
        # Ensure start <= end
        if self.start > self.end:
            self.start, self.end = self.end, self.start

    def __repr__(self):
        return f"Plate(side='{self.side}', start={self.start}, end={self.end})"

