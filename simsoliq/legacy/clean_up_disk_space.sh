#!/bin/bash


# pack all OUTCARs 
python /home/cat/heenen/Workspace/TOOLS/CatHelpers/tools/file_removepack_iterative.py -d \. -f OUTCAR -t 24 -m pack
if [ -f Au111_48H2O_OH_03/restart/OUTCAR.gz ]; then
    gunzip Au111_48H2O_OH_03/restart/OUTCAR.gz
if [ -f Au111_48H2O_OH_04/restart/OUTCAR.gz ]; then
    gunzip Au111_48H2O_OH_04/restart/OUTCAR.gz


# pack CHGCAR, CHG, LOCPOT in singlepoint folders
for dir in */singlepoints_wf     # list directories in the form 
do
    python /home/cat/heenen/Workspace/TOOLS/CatHelpers/tools/file_removepack_iterative.py -d $dir -f CHG -t 24 -m pack
    python /home/cat/heenen/Workspace/TOOLS/CatHelpers/tools/file_removepack_iterative.py -d $dir -f CHGCAR -t 24 -m pack
    python /home/cat/heenen/Workspace/TOOLS/CatHelpers/tools/file_removepack_iterative.py -d $dir -f LOCPOT -t 24 -m pack
done

# delete POTCAR in singlepoints
for dir in */singlepoints_nosolvent # list directories in the form 
do
    python /home/cat/heenen/Workspace/TOOLS/CatHelpers/tools/file_removepack_iterative.py -d $dir -f POTCAR -t 96 -m delete
    python /home/cat/heenen/Workspace/TOOLS/CatHelpers/tools/file_removepack_iterative.py -d $dir -f POTCAR.gz -t 96 -m delete
done

