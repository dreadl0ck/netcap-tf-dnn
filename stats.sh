#!/bin/bash

if [[ $1 == "" ]]; then
	echo "[INFO] need a path as argument"
	exit 1
fi

# remove trailing slash, if any
INPUT_DIR=${1%/}

echo "INPUT_DIR=$INPUT_DIR"

for f in $(ls $INPUT_DIR/*_labeled.csv); do

	res=${f%_labeled.csv}_RESULT.txt

	#recs=$(cat $res | grep 'rows.' | grep -o -E '[0-9]+' | tr '\n' ' ')
	recs=$(cat $res | grep 'rows.' | perl -pe 's/\x1b\[[0-9;]*m//g' | grep -o -E '[0-9]+' | tr '\n' ' ')

	size=$(du -h $f | cut -f -1)
	
	labels=$(cat $f | grep -v normal | wc -l | tr -d '[:space:]')
	# -1 because of the header line
	labels_count=$(($labels-1))

	
	time=$(cat $res | grep 'Time:')
	time_n=${time#"Exec Time: "}
	
	#score=$(cat $res | grep 'Validation score:' | grep -Eo '[+-]?[0-9]+([.][0-9]+)?')
	score=$(cat $res | grep 'Validation score:')
	score_n=${score#"Validation score: "}

	# echo "recs=$recs"
	# echo "size=$size"
	# echo "labels=$labels_count"
	# echo "time=$time_n"
	# echo "score=$score_n"

	file=${f#"$INPUT_DIR/"}

	printf "%-30s %-10s %-10s %-10s %-25s %s\n" $file $recs $size $labels_count "$time_n" $score_n
done