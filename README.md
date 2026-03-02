# E-commerce Analysis and Logistic Regression from Scratch

This project focuses on predicting negative customer reviews (is_bad_review) using the Olist e-commerce dataset. It represents my initial implementation of a predictive pipeline, designed to understand the underlying mathematics of machine learning algorithms and relational data processing without relying on high-level frameworks.

## Technical Implementation

### Data Processing (DuckDB)
Data extraction and transformation are handled by the DuckDB engine. This approach allows for complex joins and aggregations directly on CSV files:
- Efficiency: Data is processed in-memory using SQL, eliminating the need for extensive pre-processing in external tools.
- Aggregation: CTEs (Common Table Expressions) are used to join orders, payments, and reviews into a final training dataset.

### Model: Logistic Regression (from scratch)
The algorithm is implemented as a Python class LogisticRegression with an emphasis on an object-oriented approach:
- Optimization: The model utilizes Gradient Ascent to maximize the log-likelihood function. Parameters are updated in the direction of the gradient using the += operator.
- Numerical Stability: The sigmoid function includes clipping (between -250 and 250) to prevent numerical overflow during processing.
- Weighted Gradient: A weight vector is implemented to adjust the influence of individual classes on weight updates, which is critical for handling imbalanced data.

## Results and Evaluation
The current version of the model shows the following results on the test set:
- Accuracy: ~87%
- F1 Score: ~40%

### Interpretation
The high accuracy is misleading due to significant class imbalance (the majority of reviews are positive). The low F1 score indicates that the model struggles to correctly identify the minority class (negative reviews).

This project serves as my introduction to machine learning. I am currently focusing on studying methods such as advanced feature engineering and classification threshold tuning to achieve a better balance between precision and recall.

## Mathematical Background
- Update Rule: Weights are updated using the rule theta = theta + alpha * gradient, where we are performing gradient ascent on the log-likelihood function.
- Metrics: Precision, Recall, and F1 values are calculated manually from the confusion matrix (TP, TN, FP, FN) without external libraries.

## Data Schema
Visual representation of the Olist dataset relations:

![Data Schema](data_schema.png)

## Project Structure
- end_to_end_01_workcode.py: Main script containing SQL ETL, model class, and evaluation.
- requirements.txt: List of dependencies for environment replication.
- .gitignore: Configuration to exclude data and virtual environments from version control.

## Installation
pip install -r requirements.txt
python end_to_end_01_workcode.py
