# E-commerce Customer Satisfaction Analysis
### Predicting delivery-driven bad reviews using logistic regression built from scratch

## Business Problem
In e-commerce, a single bad review can cost more than the order itself. 
This project identifies which orders are at highest risk of generating 
a negative review — before the customer writes it.

**Key finding:** Delivery delay is the dominant driver of bad reviews, 
with 6x stronger signal than order value and 8x stronger than payment 
installments. Logistics quality, not price, determines customer satisfaction.

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 91.2% |
| Precision | 58.8% |
| Recall | 32.5% |
| F1 Score | 41.9% |

*Note: Class imbalance (89% positive reviews) makes accuracy misleading — 
F1 score is the relevant metric for minority class detection.*

## Technical Approach
- **Data pipeline:** DuckDB in-memory SQL with CTEs joining orders, 
payments and reviews across 100k+ transactions
- **Model:** Logistic regression implemented from scratch in Python — 
gradient ascent on log-likelihood, weighted gradient for class imbalance, 
sigmoid numerical stability via clipping
- **No sklearn** — mathematics implemented manually to demonstrate 
understanding of underlying algorithms

## What the Model Learned
| Feature | Weight | Interpretation |
|---------|--------|----------------|
| Delivery delay | 0.7604 | Strongest signal by far |
| Order value | 0.1288 | Expensive disappointments hurt more |
| Installments | 0.0891 | Marginal effect |

## Stack
Python · DuckDB · NumPy · Pandas

## Run
pip install -r requirements.txt
python end_to_end_01_workcode.py
