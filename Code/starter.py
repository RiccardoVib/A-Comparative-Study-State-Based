from Training import train

#data_dir = '../../Files/'
data_dir = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/JournalVA/Pickles/new/'

epochs = [1, 60]
units = 8
b_size = 600
dataset = 'CL1B'
lr = 3e-4


model = 'S4D'

train(data_dir=data_dir,
      save_folder=model+dataset,
      dataset=dataset,
      b_size=b_size,
      learning_rate=lr,
      units=units,
      epochs=epochs,
      model=model,
      inference=True)