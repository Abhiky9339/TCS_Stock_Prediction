# TCS_Stock_Prediction

# Domain : Stock Market

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


## Exploratory Data Analysis (EDA)

1. **Missing Values**: Check and handle missing values.
2. **Feature Selection**: Select relevant features (`Open`, `High`, `Low`, `Volume`) and target (`Close`).
3. **Visualization**: Use heatmaps and pairplots to visualize feature relationships and correlations.

## Model Training and Evaluation

- **Model**: Ridge Regression
- **Evaluation Metrics**: Mean Squared Error (MSE) and R-squared Score

The script trains the Ridge Regression model on scaled features and evaluates its performance using MSE and R-squared metrics.

