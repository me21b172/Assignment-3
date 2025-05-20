# DA6401_ME21B172_Assignment-3: Sequence-to-Sequence Transliteration using RNNs
---
##  Problem Statement

The task is to build a character-level sequence-to-sequence transliteration model that converts Latin-script Hindi words (e.g., "ghar") into native Devanagari script (e.g., "घर"). This involves training an encoder-decoder architecture using RNNs on the **Dakshina dataset**.
## Dataset Used

**Dakshina Dataset v1.0**  
- Source: [Google Research Datasets](https://github.com/google-research-datasets/dakshina)
- We use:
  - `hi.translit.sampled.train.tsv` for training
  - `hi.translit.sampled.test.tsv` for evaluation

Each line in the dataset contains:  
- Column 1: The Hindi word in Devanagari (target)  
- Column 2: The Latin-script transliteration (input)  
- Column 3: Frequency count (ignored in this task)

---

## Model Overview

This is a character-level sequence-to-sequence model that learns to map Latin-script Hindi words to their corresponding Devanagari representation. The architecture consists of:
- **Encoder**: An RNN-based model that processes the input sequence.
- **Decoder**: An RNN-based model that generates the output sequence.
- **Attention Mechanism** (optional): Helps the model focus on specific parts of the input sequence while generating each token of the output.

 Key features:
- Flexible RNN cell types: **RNN**, **GRU**, or **LSTM**
- Optionally **bidirectional** encoder
- **Teacher forcing** for improved learning
- Trained using **CrossEntropyLoss** with padding mask
- Evaluated using **word-level accuracy**
- Optionally visualizes **attention weights** (if attention is included)

---
## Files:
### `model.py`

* Contains Encoder, Decoder models (where modeluar functionality to change model parameters)
* Also contains a model involving attention
### `train.py`

* Contains functions to train and evaluate the model
* This file is responsible for the creation of the predictions CSV files
### `wandb_runner.py`

* Contains helper functions to run the code in wandb or normal mode
### `utils.py`

* This file is responsible for the creation of datasets (which will be eventually be used by dataloaders)
### Wandb Report Link:
https://api.wandb.ai/links/me21b172-indian-institute-of-technology-madras/2w3ailka

## How to Run
Just run following command 
```
python3 wandb_runner.py
```
Refer the code to change the arguments to train model.
