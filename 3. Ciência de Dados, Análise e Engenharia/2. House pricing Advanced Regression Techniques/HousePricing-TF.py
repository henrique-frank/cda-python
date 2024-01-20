import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_file_path = "/content/drive/MyDrive/Curso/Code/3. Ciência de Dados, Análise e Engenharia/house-prices-advanced-regression-techniques/train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))

dataset_df.head(3)

dataset_df = dataset_df.drop('Id', axis=1)
dataset_df.head(3)

print(dataset_df['SalesPrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(dataset_df['SalesPrice'], color='g', bins=100, hist_kws={'alpha': 0.4})

import numpy as np

def split_dataset(dataset, test_ratio=0.30):
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train_ds_pd, valid_ds_pd = split_dataset(dataset_df)
print("{} examples in training, {} examples in testing.".format(
    len(train_ds_pd), len(valid_ds_pd)))

label = 'SalesPrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)
valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)


tfdf.keras.get_all_models()

rf = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"]) # Optional, you can use this to include a list of eval metrics

rf.fit(x=train_ds)

tfdf.model_plotter.plot_model_in_colab(rf, tree_idx=0, max_depth=3)