#Sowing Success: How Machine Learning Helps Farmers Select the Best Crops


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("C:/Users/kalai/Downloads/soil_measures.csv")

#print(crops)
features_dict = {}
best_predictive_feature = {}
X = crops.iloc[:,:-1].values
y = crops.iloc[:, -1].values
for features in ["N", "P", "K", "ph"]:
    X_train, X_test, y_train, y_test = train_test_split(crops[features].values.reshape(-1,1), y, test_size = 0.3, random_state = 42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc= metrics.accuracy_score(y_pred, y_test)
    features_dict[features] = acc
print(features_dict)
feature_dict_sorted = sorted(features_dict.items(), key = lambda x: x[1], reverse = True)
best_predictive_feature = {feature_dict_sorted[0][0] : feature_dict_sorted[0][1]}
print("The best feature is", list(best_predictive_feature.keys())[0], "and its accuracy is", list(best_predictive_feature.values())[0])
