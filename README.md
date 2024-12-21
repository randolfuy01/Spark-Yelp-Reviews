"""
# **Spark-Yelp-Reviews**

Logistic and Random Forest Regression on Yelp Data using Apache Spark

[![Dataset Link](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)

---

## **Overview**
This project involves analyzing and modeling Yelp data to predict business performance, customer sentiments, or other key metrics. The dataset is large-scale, requiring distributed computing capabilities provided by **Apache Spark** for data preprocessing and machine learning tasks.

---

## **Objectives**
1. Preprocess Yelp's JSON data and convert it into more efficient formats like **Parquet** and **CSV**.
2. Perform exploratory data analysis (EDA) to gain insights into user reviews, business categories, and trends.
3. Build and evaluate machine learning models:
   - **Logistic Regression** for binary classification tasks (e.g., predicting positive vs. negative reviews).
   - **Random Forest Regression** for predicting continuous variables (e.g., business ratings).
4. Leverage Spark's distributed computing power to handle large-scale data efficiently.

---

## **Dataset Description**
The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset) and consists of the following files:
- **`business.json`**: Information about local businesses.
- **`review.json`**: Customer reviews of businesses.
- **`user.json`**: User profile data.
- **`checkin.json`**: Check-in information for businesses.
- **`tip.json`**: Short tips left by customers.

### **Dataset Features**
- **Business Data**:
  - `business_id`: Unique identifier for each business.
  - `name`: Name of the business.
  - `categories`: Categories associated with the business.
  - `stars`: Average rating of the business.
  - `review_count`: Total number of reviews for the business.
- **Review Data**:
  - `review_id`: Unique identifier for each review.
  - `user_id`: Identifier for the reviewer.
  - `business_id`: Identifier for the reviewed business.
  - `stars`: Rating given by the reviewer.
  - `text`: Content of the review.
  - `date`: Date of the review.

---

## **Pipeline**
### **1. Data Preprocessing**
- Read and process JSON files using **PySpark**.
- Clean and filter the dataset to remove irrelevant data.
- Convert JSON data to **Parquet** format for efficient storage and processing.
- Perform transformations to create meaningful features for modeling.

### **2. Exploratory Data Analysis (EDA)**
- Analyze distributions of ratings, review counts, and business categories.
- Visualize key insights using Python libraries like **Matplotlib** and **Seaborn**.

### **3. Machine Learning Models**
- Use Spark's MLlib library to implement:
  - **Logistic Regression**:
    - Predict whether a review is positive (4–5 stars) or negative (1–2 stars).
    - Evaluate using metrics like accuracy and F1-score.
  - **Random Forest Regression**:
    - Predict the average star rating of a business based on reviews and other features.
    - Evaluate using RMSE and R².

---

## **Technologies Used**
- **Apache Spark**: Distributed data processing and machine learning.
- **PySpark**: Python API for Spark.
- **Python**: For data visualization and additional analysis.
- **Pandas**: For small-scale data manipulation during EDA.
- **Matplotlib & Seaborn**: Data visualization libraries.
- **Kaggle Dataset**: Yelp Academic Dataset.

---

## **Setup Instructions**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Spark-Yelp-Reviews.git
   cd Spark-Yelp-Reviews
2. **Install dependencies:**
    ``` bash
    pip3 install -r requirements
