#!/bin/bash

NUM=4

echo "[INFO] running experiment $NUM"
echo "[INFO] parsing data"
netcap -r Tuesday-WorkingHours.pcap -out Tuesday-WorkingHours-$NUM
netcap -r Wednesday-WorkingHours.pcap -out Wednesday-WorkingHours-$NUM
netcap -r Thursday-WorkingHours.pcap -out Thursday-WorkingHours-$NUM
netcap -r Friday-WorkingHours.pcap -out Friday-WorkingHours-$NUM

echo "[INFO] labeling data"
netlabel -r Tuesday-WorkingHours.pcap -out Tuesday-WorkingHours-$NUM -description
netlabel -r Wednesday-WorkingHours.pcap -out Wednesday-WorkingHours-$NUM -description
netlabel -r Thursday-WorkingHours.pcap -out Thursday-WorkingHours-$NUM -description
netlabel -r Friday-WorkingHours.pcap -out Friday-WorkingHours-$NUM -description

echo "[INFO] evaluating"
eval.sh Tuesday-WorkingHours-$NUM -drop=SrcIP,DstIP
eval.sh Wednesday-WorkingHours-$NUM -drop=SrcIP,DstIP
eval.sh Thursday-WorkingHours-$NUM -drop=SrcIP,DstIP
eval.sh Friday-WorkingHours-$NUM -drop=SrcIP,DstIP

echo "[INFO] stats"
stats.sh Tuesday-WorkingHours-$NUM
stats.sh Wednesday-WorkingHours-$NUM
stats.sh Thursday-WorkingHours-$NUM
stats.sh Friday-WorkingHours-$NUM

echo "[INFO] done."