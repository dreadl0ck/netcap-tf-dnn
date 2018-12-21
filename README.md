# Netcap Tensorflow Deep Neural Network


This repository contains a python implementation for using a Deep Neural Network with [Keras](https://keras.io) and [Tensorflow](https://www.tensorflow.org),
that operates on CSV data produced by the [netcap](github.com/dreadl0ck/netcap) framework.

It is based on the implementation demonstrated by Prof Jeff Heaton's Washington University (in St. Louis) Course T81-558: Applications of Deep Neural Networks,
that has been adapted and parameterized in order to offer flexibility for experiments.

This project was created for my bachelor thesis *"Implementation and evaluation of secure and scalable anomaly-based network intrusion detection"*,
to conduct a series of experiments on identifying malicious behavior in the [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) dataset.
The thesis and presentation slides are available on [researchgate](https://www.researchgate.net/project/Anomaly-based-Network-Security-Monitoring).
Each experiment is executed with a dedicated shell script.

Watch a quick demo of the deep neural network for classification of malicious behavior, on a small PCAP dump file with traffic from the LOKI Bot.
First, the PCAP file is parsed with [netcap](github.com/dreadl0ck/netcap),
in order to get audit records that will be labeled afterwards with the [netlabel](https://github.com/dreadl0ck/netcap#netlabel-command-line-tool) tool.
The labeled CSV data for the TCP audit record type is then used for training (75%) and evaluation (25%) of the classification accuracy provided by the deep neural network.

[![asciicast](https://asciinema.org/a/217944.svg)](https://asciinema.org/a/217944)

## Usage

    $ netcap-tf-dnn.py -h
    usage: netcap-tf-dnn.py [-h] -read READ [-drop DROP] [-sample [SAMPLE]]
                            [-dropna] [-string_dummy] [-string_index]
                            [-test_size TEST_SIZE] [-loss LOSS]
                            [-optimizer OPTIMIZER]

    NETCAP compatible implementation of Network Anomaly Detection with a Deep
    Neural Network and TensorFlow

    optional arguments:
    -h, --help            show this help message and exit
    -read READ            Labeled input CSV file to read from (required)
    -drop DROP            optionally drop specified columns, supply multiple
                            with comma
    -sample [SAMPLE]      optionally sample only a fraction of records
    -dropna               drop rows with missing values
    -string_dummy         encode strings as dummy variables
    -string_index         encode strings as indices (default)
    -test_size TEST_SIZE  specify size of the test data in percent (default:
                            0.25)
    -loss LOSS            set function (default: categorical_crossentropy)
    -optimizer OPTIMIZER  set optimizer (default: adam)

## License

Apache License 2.0