import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
data = pd.read_csv('data.csv')

# Features & targets
X = data.iloc[:, :-3]  # Features (take all X except the last 3)
y_cols = ['Q69', 'Q78B', 'Q72B']  # Target columns

# One-hot encode features
encoder = OneHotEncoder()
X = encoder.fit_transform(X)

# Split data (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(X, data[y_cols], test_size=0.25, random_state=42)

for target in y_cols:
    # Label encode target
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train[target])
    y_test_encoded = label_encoder.transform(y_test[target])
    # Logistic regression
    logreg = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                                intercept_scaling=1, l1_ratio=None, max_iter=1000,
                                n_jobs=None, penalty='l2',
                                random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                                warm_start=False)
    logreg.fit(X_train, y_train_encoded)
    y_pred = logreg.predict(X_test)
    least_influence = np.abs(logreg.coef_[0, :]).argsort()[:10][::-1]
    coef_dict = {}
    coef_list = []
    for coef, feat in zip(logreg.coef_[0, :], data):
        coef_dict[feat] = round(coef, 3)
        coef_list.append((feat, round(coef, 3)))
    # Print
    print(target)
    print('Bias:', round(float(str(logreg.intercept_).replace('[', '').replace(']', '')), 3))
    print('MSE:', round(mean_squared_error(y_test_encoded, y_pred), 5))
    print('Confusion matrix:\n', confusion_matrix(y_test_encoded, y_pred))
    print('Accuracy:', round(accuracy_score(y_test_encoded, y_pred)*100, 2))
    print('Feature weights:\n', str(coef_dict).replace('{', '').replace('}', '').replace(',', '\n'), '\n')






