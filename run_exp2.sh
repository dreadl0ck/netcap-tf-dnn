#!/bin/bash

NUM=2

echo "[INFO] running experiment $NUM"
echo "[INFO] parsing data"
netcap -r Tuesday-WorkingHours.pcap -out Tuesday-WorkingHours-$NUM
netcap -r Wednesday-WorkingHours.pcap -out Wednesday-WorkingHours-$NUM
netcap -r Thursday-WorkingHours.pcap -out Thursday-WorkingHours-$NUM
netcap -r Friday-WorkingHours.pcap -out Friday-WorkingHours-$NUM

echo "[INFO] labeling data"
netcap -r Tuesday-WorkingHours.pcap -out Tuesday-WorkingHours-$NUM -label
netcap -r Wednesday-WorkingHours.pcap -out Wednesday-WorkingHours-$NUM -label
netcap -r Thursday-WorkingHours.pcap -out Thursday-WorkingHours-$NUM -label
netcap -r Friday-WorkingHours.pcap -out Friday-WorkingHours-$NUM -label

echo "[INFO] evaluating"
eval.sh Tuesday-WorkingHours-$NUM
eval.sh Wednesday-WorkingHours-$NUM
eval.sh Thursday-WorkingHours-$NUM
eval.sh Friday-WorkingHours-$NUM

echo "[INFO] stats"
stats.sh Tuesday-WorkingHours-$NUM
stats.sh Wednesday-WorkingHours-$NUM
stats.sh Thursday-WorkingHours-$NUM
stats.sh Friday-WorkingHours-$NUM

echo "[INFO] done."