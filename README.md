# Customer Churn Prediction using Machine Learning

## Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Predicting which customers are likely to leave helps organizations take proactive measures to improve customer retention.

This project develops a machine learning model to predict customer churn using historical customer data, including demographic information, account details, and service usage patterns. Multiple classification algorithms are trained and evaluated to identify the best-performing model.

---

## Objective

The objective of this project is to build a machine learning model that can accurately predict whether a customer will churn based on historical customer data.

---

## Dataset

The project uses a customer churn dataset containing information such as:

* Customer demographics
* Account information
* Subscription details
* Service usage
* Billing information
* Customer churn status (Target Variable)

**Target Variable:**

* Churn = Yes (Customer leaves)
* Churn = No (Customer stays)

---

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

---

## Machine Learning Algorithms

The following classification algorithms were implemented and compared:

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier

The model with the highest accuracy was selected as the final prediction model.

---

## Project Workflow

1. Import required libraries
2. Load the dataset
3. Perform data cleaning and preprocessing
4. Encode categorical features
5. Scale numerical features
6. Split data into training and testing sets
7. Train multiple machine learning models
8. Evaluate model performance
9. Compare model accuracies
10. Save the best-performing model
11. Predict customer churn for new data

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy Score
* Classification Report
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## Project Structure

```
Customer-Churn-Prediction/
│
├── dataset/
│   └── churn.csv
│
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb
│
├── models/
│   └── best_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── model_comparison.png
│
├── README.md
└── requirements.txt
```

---

## Results

* Successfully trained multiple machine learning classification models.
* Compared the performance of Logistic Regression, Random Forest, and Gradient Boosting.
* Selected the best-performing model based on evaluation metrics.
* Built a predictive system capable of identifying customers who are likely to churn.

---

## Future Improvements

* Hyperparameter tuning for improved accuracy.
* Deploy the model using Flask or Streamlit.
* Add more advanced ensemble learning techniques.
* Integrate real-time customer prediction.
* Develop an interactive dashboard for visualization.

---

## Conclusion

This project demonstrates the complete machine learning workflow for solving a real-world customer churn prediction problem. By leveraging historical customer data and classification algorithms, businesses can identify at-risk customers and implement strategies to improve customer retention.

---

## Author

**Hari Prasad S N**

Machine Learning Virtual Internship – CodSoft

## License
This project is developed for educational purposes as part of the CodSoft Machine Learning Internship.
