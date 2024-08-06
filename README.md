# TCS_Stock_Prediction

Certainly! Below is a `README.md` file that explains how to set up and run a stock prediction application using Ridge Regression, including all stages of Exploratory Data Analysis (EDA).

---

# TCS Stock Prediction Using Ridge Regression

This project demonstrates how to predict TCS (Tata Consultancy Services) stock prices using Ridge Regression. The process includes all stages of Exploratory Data Analysis (EDA) and data preparation.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Data](#data)
- [Setup](#setup)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Usage](#usage)
- [License](#license)

## Project Overview

This project uses historical stock data of TCS to predict the closing price of the stock. The Ridge Regression algorithm is employed for prediction after performing comprehensive Exploratory Data Analysis (EDA) to understand the data and prepare it for modeling.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data

The dataset used in this project is assumed to be in a CSV file named `tcs_stock_data.csv`. This file should contain historical stock data with columns like `Open`, `High`, `Low`, `Volume`, and `Close`.

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/tcs-stock-prediction.git
   cd tcs-stock-prediction
   ```

2. **Place the Dataset**

   Ensure that `tcs_stock_data.csv` is in the root directory of the project.

3. **Run the Script**

   Create a Python script named `stock_prediction.py` with the following content:

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import Ridge
   from sklearn.metrics import mean_squared_error, r2_score

   # Load the dataset
   df = pd.read_csv('tcs_stock_data.csv')

   # Check for missing values
   print(df.isnull().sum())
   df.fillna(method='ffill', inplace=True)

   # Define features and target
   features = df[['Open', 'High', 'Low', 'Volume']]
   target = df['Close']

   # Data visualization
   plt.figure(figsize=(10, 6))
   sns.heatmap(df[['Open', 'High', 'Low', 'Volume', 'Close']].corr(), annot=True, cmap='coolwarm')
   plt.title('Feature Correlation Heatmap')
   plt.show()

   sns.pairplot(df[['Open', 'High', 'Low', 'Volume', 'Close']])
   plt.show()

   # Split data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

   # Feature scaling
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   # Train Ridge Regression model
   ridge_reg = Ridge(alpha=1.0)
   ridge_reg.fit(X_train_scaled, y_train)

   # Predict and evaluate
   y_pred = ridge_reg.predict(X_test_scaled)
   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)

   print(f'Mean Squared Error: {mse}')
   print(f'R-squared Score: {r2}')
   ```

   Run the script using:

   ```bash
   python stock_prediction.py
   ```

## Exploratory Data Analysis (EDA)

1. **Missing Values**: Check and handle missing values.
2. **Feature Selection**: Select relevant features (`Open`, `High`, `Low`, `Volume`) and target (`Close`).
3. **Visualization**: Use heatmaps and pairplots to visualize feature relationships and correlations.

## Model Training and Evaluation

- **Model**: Ridge Regression
- **Evaluation Metrics**: Mean Squared Error (MSE) and R-squared Score

The script trains the Ridge Regression model on scaled features and evaluates its performance using MSE and R-squared metrics.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This `README.md` provides an overview of the project, setup instructions, and details on how to run and evaluate the stock prediction model using Ridge Regression. Adjust the instructions as necessary based on your project structure and specific requirements.
