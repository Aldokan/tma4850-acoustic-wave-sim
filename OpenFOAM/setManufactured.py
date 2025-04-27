import numpy as np

filePath = "0/C"
filePathPa = "0/pa"

import numpy as np


def read_internal_field_vectors(file_path: str) -> np.ndarray:
    """
    Reads an OpenFOAM field file and extracts the internalField block
    (the numbers before the boundaryField data). The block is expected to be
    formatted as:

      internalField   nonuniform List<scalar>
      <count>
      (
      (x y z)
      (x y z)
      ...
      )
      ;

    Returns a NumPy array where each row is one vector.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line that starts the internalField block.
    internal_idx = None
    for i, line in enumerate(lines):
        if "internalField" in line:
            internal_idx = i
            break
    if internal_idx is None:
        raise ValueError("Could not find internalField in the file.")

    # After the internalField line, the next line should be the count.
    count_line = lines[internal_idx + 1].strip()
    try:
        count = int(count_line)
    except ValueError:
        print("Warning: Could not interpret the count of vectors.")
        count = None  # We'll ignore the count if it can't be read

    # Find the start of the data block (line with just "(")
    start_idx = None
    for i in range(internal_idx, len(lines)):
        if lines[i].strip() == "(":
            start_idx = i + 1  # Data starts on the next line
            break
    if start_idx is None:
        raise ValueError("Could not find the start of the internal field data ('(').")

    # Find the end of the data block (line with ")")
    end_idx = None
    for i in range(start_idx, len(lines)):
        if lines[i].strip() == ")":
            end_idx = i
            break
    if end_idx is None:
        raise ValueError("Could not find the end of the internal field data (')').")

    # Process the lines containing the vectors.
    vector_list = []
    for line in lines[start_idx:end_idx]:
        # Remove extra whitespace.
        line_clean = line.strip()
        # Expect each line to be of the form (x y z). Remove the surrounding parentheses.
        if line_clean.startswith("(") and line_clean.endswith(")"):
            line_clean = line_clean[1:-1].strip()
        # Split the line into tokens and convert each to a float.
        try:
            vector = [float(token) for token in line_clean.split()]
        except ValueError as e:
            # Skip lines that cannot be parsed into floats.
            continue
        vector_list.append(vector)

    if count is not None and len(vector_list) != count:
        print(f"Warning: Expected {count} vectors, but parsed {len(vector_list)}.")

    return np.array(vector_list)


# Example usage:
internal_vectors = read_internal_field_vectors(filePath)
#print("Internal field vectors:\n", internal_vectors)

x = np.array([i[0] for i in internal_vectors])
y = np.array([i[1] for i in internal_vectors])


def function(x: np.array,y: np.array) -> np.array:
    x0, y0 = 4, 3
    p0 = 22440
    sigma = 0.25

    return p0 * np.exp(-((x - x0)**2 + (y - y0)**2) / (2*sigma**2)) + 101325

pa = function(x,y)
n = len(pa)


def write_field_file(file_path: str, n: int, interpValues) -> None:
    """
      n
      (
      scalarValue
      )
      ;
    """
    with open(file_path, 'r+') as file:
        lines = file.readlines()

        # Look for the target line and replace it.
        targetLine = "internalField   uniform 101325;"
        insertIndex = None
        for index, line in enumerate(lines):
            if targetLine in line:
                lines[index] = "internalField   nonuniform List<scalar>\n"
                insertIndex = index + 1
                break

        if insertIndex is None:
            print("Target line not found in file.")
            return

        # Insert header lines for the new block.
        lines.insert(insertIndex, f"{n}\n")
        lines.insert(insertIndex + 1, "(\n")

        # Insert the scalar field values.
        currentIndex = insertIndex + 2
        for value in interpValues:
            lines.insert(currentIndex, f"{value}\n")
            currentIndex += 1

        # Insert the closing bracket and semicolon.
        lines.insert(currentIndex, ")\n")
        lines.insert(currentIndex + 1, ";\n")

        # Write the modified content back to the file.
        file.seek(0)
        file.writelines(lines)
        file.truncate()


write_field_file(filePathPa, n, pa)
