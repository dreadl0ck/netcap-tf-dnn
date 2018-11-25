#!/bin/bash

NUM=1

echo "[INFO] running experiment $NUM"
echo "[INFO] parsing data"
netcap -r Tuesday-WorkingHours.pcap -out Tuesday-WorkingHours-$NUM
netcap -r Wednesday-WorkingHours.pcap -out Wednesday-WorkingHours-$NUM
netcap -r Thursday-WorkingHours.pcap -out Thursday-WorkingHours-$NUM
netcap -r Friday-WorkingHours.pcap -out Friday-WorkingHours-$NUM

echo "[INFO] labeling data"
netlabel -r Tuesday-WorkingHours.pcap -out Tuesday-WorkingHours-$NUM
netlabel -r Wednesday-WorkingHours.pcap -out Wednesday-WorkingHours-$NUM
netlabel -r Thursday-WorkingHours.pcap -out Thursday-WorkingHours-$NUM
netlabel -r Friday-WorkingHours.pcap -out Friday-WorkingHours-$NUM

echo "[INFO] evaluating"
eval.sh Tuesday-WorkingHours-$NUM -string_dummy
eval.sh Wednesday-WorkingHours-$NUM -string_dummy
eval.sh Thursday-WorkingHours-$NUM -string_dummy
eval.sh Friday-WorkingHours-$NUM -string_dummy

echo "[INFO] stats"
stats.sh Tuesday-WorkingHours-$NUM
stats.sh Wednesday-WorkingHours-$NUM
stats.sh Thursday-WorkingHours-$NUM
stats.sh Friday-WorkingHours-$NUM

echo "[INFO] done."