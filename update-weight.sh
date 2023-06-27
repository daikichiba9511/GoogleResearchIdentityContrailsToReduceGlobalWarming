#!/bin/bash
set -e

SRC_NAME="exp003"

ls "./output/${SRC_NAME}"
echo "Version: ${SRC_NAME}" >./output/sub/version.txt
cp -R ./output/${SRC_NAME}/${SRC_NAME}-*.pth ./output/sub/

kaggle datasets version -p ./output/sub -m "update: ${SRC_NAME}"