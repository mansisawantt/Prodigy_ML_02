# Prodigy_ML_02
---

# Customer Segmentation using Machine Learning

## Project Aim

This project aims to perform customer segmentation using clustering techniques to identify distinct customer groups based on purchasing behavior. This helps businesses tailor their marketing strategies effectively.

## Table of Contents

1. [Project Overview](#overview)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Analysis](#analysis)
6. [Results](#results)
7. [Contributing](#contributing)
8. [Acknowledgements](#acknowledgements)

## Dataset

The dataset used in this project is available on Kaggle:  
[Customer Segmentation Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

It contains customer data with features such as:
- **Customer ID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Annual Income**: Customer's yearly income.
- **Spending Score**: Score assigned based on customer spending patterns.

## Installation

To set up the environment, install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Load the Dataset**:
   Import the dataset into a Pandas DataFrame and inspect its structure.

2. **Exploratory Data Analysis (EDA)**:
   - Analyze distributions of features.
   - Identify correlations between customer attributes.

3. **Data Preprocessing**:
   - Handle missing values (if any).
   - Scale numerical features for clustering.

4. **Customer Segmentation using Clustering**:
   - Apply K-Means clustering to segment customers.
   - Evaluate optimal clusters using the Elbow Method.

5. **Visualization of Segments**:
   - Use scatter plots and pair plots to visualize customer segments.

### Example Usage

```python
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
```

Ensure that the dataset file (`Mall_Customers.csv`) is in the correct directory path for loading.

## Analysis

### Exploratory Data Analysis
- Distribution of age, income, and spending score.
- Identifying potential patterns in customer spending behavior.

### Clustering Techniques
- **K-Means Clustering**: Used to group customers based on their spending and income levels.
- **Elbow Method**: Determines the optimal number of clusters.

## Results

Key insights from the analysis:
- **Segmented Customers**: Identified distinct groups based on purchasing patterns.
- **Targeted Marketing**: Businesses can use this segmentation for personalized marketing strategies.
- **Data Visualization**: Showcases clusters with clear separations for actionable insights.

## Acknowledgements

Thanks to the following libraries and tools used in this project:
- [Pandas](https://pandas.pydata.org/) - Data manipulation.
- [NumPy](https://numpy.org/) - Numerical computing.
- [Matplotlib](https://matplotlib.org/) - Data visualization.
- [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization.
- [Scikit-learn](https://scikit-learn.org/) - Machine learning models and evaluation.

---


