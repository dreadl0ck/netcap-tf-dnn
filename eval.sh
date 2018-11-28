#!/bin/bash

if [[ $1 == "" ]]; then
	echo "[ERROR] need a path as argument"
	exit 1
fi

for f in $(ls $1/*_labeled.csv); do
	echo "[INFO] processing $f"
	# "${@:2}" passes down all arguments after the directory name
	netcap-tf-dnn.py -read $f "${@:2}" | tee "${f%_labeled.csv}_RESULT.txt"
done