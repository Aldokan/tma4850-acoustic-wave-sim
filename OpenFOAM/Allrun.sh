#!/bin/bash
read -r -d '' INPUT_PARAMS << EOM
0

EOM
PYTHON_SCRIPT="VerticeMakerBlockMesh/main.py"
python3 "$PYTHON_SCRIPT" <<< "$INPUT_PARAMS" > system/blockMeshDict

rm -r 0 
foamListTimes -rm
cp -r 0.orig 0

blockMesh
echo "blockMeshDict updated and blockMesh executed successfully."

#setFields
acousticFoam -postProcess -func writeCellCentres
python3 setManufactured.py
acousticFoam
#foamToVTK
