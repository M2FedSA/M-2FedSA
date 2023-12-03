# M-2FedSA
Implementation of M^2FedSA
# M^2FedSA: Multimodal Federated Learning via Model Splitting and Adapting


 This repository contains the official code for our proposed method, HAMFL, and the experiments in our paper: [Adaptive Hyper-graph Aggregation for Modality-Agnostic Federated Learning](https://openreview.net/forum?id=odSc7goxps&noteId=odSc7goxps)

 < img src="HAMFL.png" width="720" height="460" />

 ## Dependencies

The code requires Python >= 3.6 and PyTorch >= 1.2.0. To install the other dependencies:

        `pip install -r requirements.txt`.

## Data

This code uses the [EPIC-Kitchens](https://epic-kitchens.github.io/2023), [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and [MELD](https://affective-meld.github.io/) dataset.

Notice : All three datasets need to be swapped out into a public dataset.

## Usage
Voting after Clustering is run using a command of the following form:

       `python main.py` or `python -u main.py > log.txt` to save the log file

A full list of configuration parameters and their descriptions are given in `config.py`.


<!-- ## Citation

Please cite our paper if you use our implementation of Voting after Clustering: -->
