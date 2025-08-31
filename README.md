
# 360-Degree Customer Intelligence Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Pandas%20%7C%20Scikit--learn%20%7C%20mlxtend-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Project Overview
This project develops a comprehensive, 4-in-1 customer intelligence solution using a single e-commerce dataset. The goal is to demonstrate a wide range of data science techniques to solve interconnected, real-world business problems, moving from understanding customers to predicting their behavior and taking action.

## 📊 Dataset
The model was trained on the "Online Retail II UCI" dataset, which contains transactional data for a UK-based online retailer.
* **Dataset Link**: [https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

## 💡 4 Business Problems Solved
This single project tackles four key business challenges in a logical sequence:

1.  **Customer Segmentation**: Used **RFM Analysis** and **K-Means Clustering** to group customers into distinct, actionable personas like "Champions," "At-Risk Customers," and "New Customers."

2.  **Customer Churn Prediction**: Built a **Logistic Regression** model to predict the probability of a customer churning (i.e., becoming inactive) based on their behavioral segment and RFM scores.

3.  **Customer Lifetime Value (CLV) Prediction**: Developed a **Random Forest Regressor** to predict the future revenue a customer is likely to generate in the next 90 days.

4.  **Product Recommendation**: Created a simple recommender system using **Association Rule Mining (Apriori)** to find "if you buy this, you'll also like that" product associations from recent transaction data.

## 🎛️ Interactive Dashboard
An interactive, tabbed dashboard was built directly in the notebook using `ipywidgets` to demonstrate all four solutions in real-time, allowing for on-the-fly predictions and analysis.

## 💻 Technology Stack
* **Python**
* **Data Manipulation**: Pandas, NumPy
* **Machine Learning**: Scikit-learn, mlxtend
* **Visualization**: Matplotlib, Seaborn
* **Interactive UI**: ipywidgets
* **Environment**: Jupyter Notebook, Git

## ⚙️ How to Run
1.  Clone this repository.
2.  Install the dependencies from the `requirements.txt` file.
3.  The main analysis, model training, and interactive UI are contained in the `.ipynb` notebook file.
