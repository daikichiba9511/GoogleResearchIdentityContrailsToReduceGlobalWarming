#!/bin/bash
set -e

# SRC_NAME="exp003" # 6/27
# SRC_NAME="exp004" # 6/28
# SRC_NAME="exp005" # 6/29
SRC_NAME="exp008_1" # 7/1

ls "./output/${SRC_NAME}"
rm -rf ./output/sub/*.pth
ls "./output/sub"
echo "Version: ${SRC_NAME}" >./output/sub/version.txt
cp -R ./output/${SRC_NAME}/${SRC_NAME}-*.pth ./output/sub/

kaggle datasets version -p ./output/sub -m "update: ${SRC_NAME}"
