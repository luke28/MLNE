#!/bin/bash
# 批量运行一个文件夹下的所有conf，输入参数为conf文件夹路径或其中的一个子文件夹路径

MAIN_PATH=$(cd `dirname $0`; pwd)
echo $MAIN_PATH

path=$1
echo $path
pre=""
if [ $(basename $path) != "conf" ]; then
    pre=$(basename $path)/
fi

log_path=$MAIN_PATH"/../tmp/"$pre
echo $log_path

if [ ! -d "$log_path" ]; then
    mkdir -p "$log_path"
fi
files=$(ls $path)
for filename in $files
do
    echo $pre${filename%.*}
    python $MAIN_PATH/main.py --conf $pre${filename%.*} > $log_path${filename%.*}.log
done

