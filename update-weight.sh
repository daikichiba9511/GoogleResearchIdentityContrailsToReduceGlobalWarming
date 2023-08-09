#!/bin/bash
set -e

# SRC_NAME="exp003" # 6/27
# SRC_NAME="exp004" # 6/28
# SRC_NAME="exp005" # 6/29
# SRC_NAME="exp008_1" # 7/1
# SRC_NAME="exp009_1" # 7/8

# psuedo label
# SRC_NAME1="exp020" # 8/1
# SRC_NAME2="exp021" # 8/1

# TTA+tf-effv2_s
SRC_NAME1="exp027" # 8/6
SRC_NAME2="exp028" # 8/7
SRC_NAME2="exp030" # 8/7

rm -rf ./output/sub/*.pth
ls "./output/sub"
rm "./output/sub/version.txt"

LOG_MSG="update: "

for src_name in $SRC_NAME1 $SRC_NAME2; do
	echo " ######## Source: ${src_name} ######### "
	ls "./output/${src_name}"
	echo "Version: ${src_name}" >>./output/sub/version.txt
	mkdir -p "./output/sub/${src_name}"
	cp -R ./output/${src_name}/${src_name}-*.pth ./output/sub/${src_name}/
	ls "./output/sub/${src_name}"
	LOG_MSG="${LOG_MSG} ${src_name},"
done

echo "## $LOG_MSG ##"
kaggle datasets version --dir-mode zip -p ./output/sub/ -m "update: ${LOG_MSG}"
