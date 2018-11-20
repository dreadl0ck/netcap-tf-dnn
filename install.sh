#!/bin/bash

INSTALL_PATH="/usr/local/bin"

cp netcap-tf-dnn.py $INSTALL_PATH
cp eval.sh $INSTALL_PATH
cp run_exp*.sh $INSTALL_PATH
cp stats.sh $INSTALL_PATH

chmod +x /usr/local/bin/*.sh
chmod +x /usr/local/bin/netcap-tf-dnn.py