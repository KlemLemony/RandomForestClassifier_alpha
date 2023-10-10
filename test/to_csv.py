import pandas as pd

predictions = ['dsfsaf', 'sdfsafas', 'w', 'y', 'i']

import os

current_directory = os.getcwd()
print("Текущая директория:", current_directory)

df_predictions = pd.DataFrame({'Id': [i for i in range(0,5)],  'Predictions': [' '.join(list(map(str, pred))) for pred in predictions]})

df_predictions.to_csv("predictions.csv", index=False)
