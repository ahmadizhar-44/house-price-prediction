import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Size (sq ft)': [850, 900, 1200, 1500, 1800, 2000, 2200, 2500],
    'Price ($)': [100000, 120000, 150000, 180000, 200000, 220000, 250000, 280000]
}

df = pd.DataFrame(data)

X = df[['Size (sq ft)']]
y = df['Price ($)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

size = [[1700]]
predicted_price = model.predict(size)
print(f"\nPredicted Price for 1700 sq ft: ${predicted_price[0]:.2f}")

plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($)")
plt.title("House Price Prediction")
plt.legend()
plt.show()