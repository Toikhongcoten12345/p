import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

#
from sklearn.model_selection import train_test_split

df = pd.read_csv("predict_Covid_dangerous_areas_covid19.csv")
from sklearn.ensemble import RandomForestClassifier


x = df.drop('Correlate', axis=1)
y = df['Correlate']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Khởi tạo và huấn luyện mô hình RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

# Dự đoán trên tập kiểm tra
rf_pred = rf.predict(x_test)
print("Dự đoán từ RandomForestClassifier: ", rf_pred)

# lưu Model
import pickle
# Lưu mô hình Random Forest Classifier
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf, file)