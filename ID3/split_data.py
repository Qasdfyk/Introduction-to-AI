import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('cardio_train.csv', sep=';')
age_bins = [0, 5*365, 10*365, 15*365, 20*365, 25*365, 30*365, 35*365, 40*365, 45*365, 50*365, 55*365, 60*365, 65*365, 70*365]
age_labels = [int((age_bins[i] + age_bins[i+1]) / 2) for i in range(len(age_bins)-1)]

weight_bins = list(range(0, 151, 10))
weight_labels = [int((weight_bins[i] + weight_bins[i+1] - 1) / 2) for i in range(len(weight_bins)-1)]

height_bins = list(range(50, 201, 10))
height_labels = [int((height_bins[i] + height_bins[i+1] - 1) / 2) for i in range(len(height_bins)-1)]

ap_hi_bins = list(range(60, 161, 5))
ap_hi_labels = [int((ap_hi_bins[i] + ap_hi_bins[i+1] - 1) / 2) for i in range(len(ap_hi_bins)-1)]

ap_lo_bins = list(range(40, 141, 5))
ap_lo_labels = [int((ap_lo_bins[i] + ap_lo_bins[i+1] - 1) / 2) for i in range(len(ap_lo_bins)-1)]


df['age'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
df['weight'] = pd.cut(df['weight'], bins=weight_bins, labels=weight_labels, right=False)
df['height'] = pd.cut(df['height'], bins=height_bins, labels=height_labels, right=False)
df['ap_hi'] = pd.cut(df['ap_hi'], bins=ap_hi_bins, labels=ap_hi_labels, right=False)
df['ap_lo'] = pd.cut(df['ap_lo'], bins=ap_lo_bins, labels=ap_lo_labels, right=False)

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

train_data.to_csv('train_data.csv', sep=';', index=False)
val_data.to_csv('val_data.csv', sep=';', index=False)
test_data.to_csv('test_data.csv', sep=';', index=False)