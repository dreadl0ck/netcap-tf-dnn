#!/bin/bash

NUM=4

echo "[INFO] running experiment $NUM"

if [ -d "Tuesday-WorkingHours" ]
then
    echo "[INFO] using netcap data"

    # make sure all folders for the current experiment exist
    mkdir Tuesday-WorkingHours-$NUM
    mkdir Wednesday-WorkingHours-$NUM
    mkdir Thursday-WorkingHours-$NUM
    mkdir Friday-WorkingHours-$NUM

    # copy netcap data from previously generated folders to the current experiment folder
    cp Tuesday-WorkingHours/*.ncap.gz Tuesday-WorkingHours-$NUM
    cp Wednesday-WorkingHours/*.ncap.gz Wednesday-WorkingHours-$NUM
    cp Thursday-WorkingHours/*.ncap.gz Thursday-WorkingHours-$NUM
    cp Friday-WorkingHours/*.ncap.gz Friday-WorkingHours-$NUM
else
    echo "[INFO] parsing data"
    netcap -r Tuesday-WorkingHours.pcap -out Tuesday-WorkingHours
    netcap -r Wednesday-WorkingHours.pcap -out Wednesday-WorkingHours
    netcap -r Thursday-WorkingHours.pcap -out Thursday-WorkingHours
    netcap -r Friday-WorkingHours.pcap -out Friday-WorkingHours
fi

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