#!/bin/bash

MAIN_PATH=$(cd `dirname $0`; pwd)

path=$1
echo $path
pre=""
if [ $(basename $path) != "conf" ]; then
   pre=$(basename $path)/
fi
files=$(ls $path)
for filename in $files
do
   echo $pre${filename%.*}
   python $MAIN_PATH/main.py --conf $pre${filename%.*}
done

