import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv("C:\\Users\Albina\Desktop\df_train.csv", sep=';')
df_test = pd.read_csv("C:\\Users\Albina\Desktop\df_test.csv", sep=';')

X = df_train['Data'].str.split(',').apply(lambda x: list(map(int, x)))
y = df_train['Target'].str.split(',').apply(lambda x: list(map(int, x)))
df_test = df_test['Data'].str.split(',').apply(lambda x: list(map(int, x)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_flat = [' '.join(map(str, sublist)) for sublist in y_train]
y_test_flat = [' '.join(map(str, sublist)) for sublist in y_test]

max_len = max(max(X_train.apply(len)), max(df_test.apply(len)))

X_train = X_train.apply(lambda x: np.pad(x, (0, max_len - len(x)), 'constant'))
df_test = df_test.apply(lambda x: np.pad(x, (0, max_len - len(x)), 'constant'))

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train.tolist(), y_train_flat)

predictions = model.predict(df_test.tolist())

predictions = [list(map(int, pred.split())) for pred in predictions]

submission_file = pd.DataFrame({'Predictions': [' '.join(map(str, pred)) for pred in predictions]})

submission_file.to_csv("predictions.csv", index=False)


'''
def mapk(y_true, y_pred, k=10):
    apk_values = []
    for i in range(len(y_true)):
        y_true_i = list(map(int, y_true[i].split()))
        y_pred_i = y_pred[i][:k]
        score = 0.0
        num_hits = 0.0
        for j in range(k):
            if y_pred_i[j] in y_true_i:
                num_hits += 1
                score += num_hits / (j + 1)
        if not y_true_i:
            apk_values.append(0.0)
        else:
            apk_values.append(score / min(len(y_true_i), k))
    return np.mean(apk_values)

mapk_score = mapk(y_test_flat, predictions, k=10)
print(f'MAP@10: {mapk_score}')
'''
