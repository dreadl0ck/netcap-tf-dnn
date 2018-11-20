#!/bin/bash

if [[ $1 == "" ]]; then
	echo "[INFO] need a path as argument"
	exit 1
fi

for f in $(ls $1/*_labeled.csv); do
	echo "[INFO] ------------" $f "------------"
	res=${f%_labeled.csv}_RESULT.txt
	echo "[INFO] Num Records: " $(cat $res | grep 'rows.')
	echo "[INFO] Size: " $(du -h $f)
	labels=$(cat $f | grep -v normal | wc -l | tr -d '[:space:]')
	# -1 because of the header line
	echo "[INFO] Labels: $(($labels-1))" 
	time=$(cat $res | grep 'Time:')
	time1=${time//".0"/""}
	time2=${time1//", "/" "}
	echo $time2
	echo $(cat $res | grep 'score:')
done