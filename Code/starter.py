from Training import train

"""
main script

"""
# data_dir: the directory in which datasets are stored
data_dir = '../../Files/'

epochs = 200
units = 8 # number of model's units
b_size = 600 # batch size

lr = 3e-4 # initial learning rate

datasets = ['CL1B', 'LA2A', 'Pultec', 'Saturator', 'FilterNeutron', 'ODNeutron', 'OD']
models = ['LSTM', 'ED', 'S4D', 'S6', 'LRU'] 

for dataset in datasets:
      for model in models:
            train(data_dir=data_dir,
                  save_folder=model+dataset,
                  dataset=dataset,
                  b_size=b_size,
                  learning_rate=lr,
                  units=units,
                  epochs=epochs,
                  model=model,
                  inference=False)
