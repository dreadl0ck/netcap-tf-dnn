#!/bin/bash

if [[ $1 == "" ]]; then
	echo "[INFO] need a path as argument"
	exit 1
fi

# remove trailing slash, if any
INPUT_DIR=${1%/}

echo -e "\nINPUT_DIR=$INPUT_DIR"

printf "%-30s %-10s %-10s %-10s %-16s %s\n" "Filename" "Records" "Size" "Labels" "Time" "Score"
for f in $(ls $INPUT_DIR/*_labeled.csv); do

	res=${f%_labeled.csv}_RESULT.txt

	recs=$(cat $res | grep 'rows\.' | perl -pe 's/\x1b\[[0-9;]*m//g' | grep -o -E '[0-9]+' | tr '\n' ' ')

	size=$(du -h $f | cut -f -1)
	
	labels=$(cat $f | grep -v normal | wc -l | tr -d '[:space:]')
	# -1 because of the header line
	labels_count=$(($labels-1))

	time=$(cat $res | grep 'Time:')
	time_n=${time#"[INFO] Exec Time: "}
	
	score=$(cat $res | grep 'Validation score:')
	score_n=${score#"[INFO] Validation score: "}

	file=${f#"$INPUT_DIR/"}
	
	# echo "file=$file "
	# echo "recs=$recs "
	# echo "size=$size "
	# echo "labels_count=$labels_count "
	# echo "time_n=$time_n"
	# echo "score_n=$score_n"

	printf "%-30s %-10s %-10s %-10s %-25s %s\n" $file $recs $size $labels_count "$time_n" $score_n
done