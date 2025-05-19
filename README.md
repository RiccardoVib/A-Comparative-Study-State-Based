# A Comparative Study of State-based Neural Networks for Virtual Analog Audio Effects Modeling

This code repository is for the article _A Comparative Study of State-based Neural Networks for Virtual Analog Audio Effects Modeling_, on review.

This repository contains all the necessary utilities to use our architectures. Find the code located inside the "./Code" folder, and the weights of pre-trained models inside the "./Weights" folder

Visit our [companion page with audio examples](https://riccardovib.github.io/Comparative_pages/)

### Contents

1. [Datasets](#datasets)
2. [How to Train and Run Inference](#how-to-train-and-run-inference)
3. [VST Download](#vst-download)

<br/>

# Datasets

Datsets are available [here](https://www.kaggle.com/datasets/riccardosimionato/audio-effects-datasets-vol-1)

Our architectures were evaluated on seven datasets: 
- OD300 Overdrive 
- Neutron's Overdrive Module
- Helper Saturator
- Universal Pultec Equalizer
- LA-2A optical compressor
- CL 1B optical compressor
- Neutron's Low-pass Filter Module

# How To Train and Run Inference 

First, install Python dependencies:
```
cd ./code
pip install -r requirements.txt
```

To train models, use the starter.py script.
Ensure you have loaded the dataset into the chosen datasets folder

Available options: 
* --model_save_dir - Folder directory in which to store the trained models [str] (default ="./models")
* --data_dir - Folder directory in which the datasets are stored [str] (default="./datasets")
* --datasets - The names of the datasets to use. [ [str] ] (default=[" "] )
* --epochs - Number of training epochs. [int] (defaut=60)
* --model - The name of the model to train ('LSTM', 'ED', 'LRU', 'S4D', 'S6') [str] (default=" ")
* --batch_size - The size of each batch [int] (default=8 )
* --hidden_layer_sizes = The hidden layer size (amount of units) of the network. [ [int] ] (default=8)
* --mini_batch_size - The mini batch size [int] (default=2048)
* --learning_rate - the initial learning rate [float] (default=3e-4)
* --only_inference - When True, skips training and runs only inference on the pre-model. When False, runs training and inference on the trained model. [bool] (default=False)
 

Example training case: 
```
cd ./code/

python starter.py --datasets OD --model LSTM --epochs 500 
```

To only run inference on an existing pre-trained model, use the "only_inference". In this case, ensure you have the existing model and dataset (to use for inference) both in their respective directories with corresponding names.

Example inference case:
```
cd ./code/
python starter.py --datasets OD --model LSTM --only_inference True
```


# VST Download

Coming soon...
