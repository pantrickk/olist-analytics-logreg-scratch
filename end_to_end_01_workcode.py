import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect(database=":memory:")

con.execute("""
    CREATE VIEW orders AS SELECT * FROM 'data/olist_orders_dataset.csv';
    CREATE VIEW payments AS SELECT * FROM 'data/olist_order_payments_dataset.csv';
    CREATE VIEW reviews AS SELECT * FROM 'data/olist_order_reviews_dataset.csv';
""")

res_total = con.execute("SELECT count(*) FROM orders").df()
print(res_total)

res_dist = con.execute("""
    SELECT 
        review_score, 
        count(*) as count 
    FROM reviews 
    GROUP BY review_score 
    ORDER BY review_score
""").df()
print(res_dist)

final_df = con.execute("""
    WITH payments_agg AS (
        SELECT 
            order_id, 
            SUM(payment_value) AS sum_value,
            MAX(payment_installments) AS max_installments
        FROM payments
        GROUP BY order_id
    ),
    reviews_agg AS (
        SELECT 
            order_id,
            review_score,
            CASE WHEN MIN(review_score) = 1 THEN 1 ELSE 0 END AS is_bad_review
        FROM reviews
        GROUP BY order_id, review_score
    )
    SELECT 
        o.order_id,
        p.max_installments,
        p.sum_value,
        date_diff('day', o.order_estimated_delivery_date, o.order_delivered_customer_date) AS delivery_delay,
        r.review_score,
        r.is_bad_review,
    FROM orders o
    LEFT JOIN payments_agg p ON o.order_id = p.order_id
    LEFT JOIN reviews_agg r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
""").df()

final_df = final_df.dropna()

test_df = con.execute("""
    SELECT 
        review_score, 
        AVG(delivery_delay) as avg_delay, 
        AVG(sum_value) as avg_price
    FROM final_df
    GROUP BY review_score
    ORDER BY review_score
""").df()

class LogisticRegression():
    def __init__(self):
        self.learning_rate = 0.005
        self.epochs = 20000
        self.weights = None

    def sigmoid(self, z):
        z_clipped = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z_clipped))
    
    def compute_log_likelihood(self, X, y):
        z = X @ self.weights
        p = self.sigmoid(z)

        epsilon = 1e-15

        log_likelihood = y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon)  

        return np.sum(log_likelihood)
    
    

    def fit(self, X, y):
        n_columns = X.shape[1]
        self.weights = np.zeros((n_columns, 1))

        n_rows = X.shape[0]
        weights_vector = np.where(y == 1, 2.5, 1)
        for i in range(self.epochs):
            z = X @ self.weights
            p = self.sigmoid(z)

            gradient = (1 / n_rows) * (X.T @ ((y - p) * weights_vector))

            self.weights += self.learning_rate * gradient

            if i % 1000 == 0:
                current_LL = self.compute_log_likelihood(X, y)

                print(f"Aktuálna epocha: {i}, s chybou: {current_LL}")

    def predict_probab(self, X):
        return self.sigmoid(X @ self.weights)
    
    def predict(self, X, threshold=0.4):
        probs = self.predict_probab(X)
        return (probs >= threshold).astype(int)

            
model = LogisticRegression()

vector_y = final_df["is_bad_review"].to_numpy().reshape(-1, 1).astype(float)
features = ["max_installments","sum_value", "delivery_delay"]
X_scaled = (final_df[features] - final_df[features].mean()) / final_df[features].std()
X_array = X_scaled.to_numpy()
ones_array = np.ones((X_array.shape[0], 1))
X_array = np.hstack([ones_array, X_array]).astype(float)

model.fit(X_array, vector_y)

predictions = model.predict(X_array)
accuracy = np.mean(predictions == vector_y)

print(f"Finálna precíznosť (accuracy) modelu: {accuracy * 100:.2f} %")

tp = np.sum((predictions == 1) & (vector_y == 1))
tn = np.sum((predictions == 0) & (vector_y == 0))
fp = np.sum((predictions == 1) & (vector_y == 0))
fn = np.sum((predictions == 0) & (vector_y == 1))

precision = tp / (tp + fp + 1e-15)
recall = tp / (tp + fn + 1e-15)

f1_score = f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)

print(f"Presnosť (precision) modelu je {precision * 100:.2f} %")
print(f"Citlivosť (sensitivity) modelu je {recall * 100:.2f} %")
print(f"F1 skóre modelu je {f1_score * 100:.2f} %")

feature_names = ["Intercept", "Splátky","Cena", "Meškanie"]
if model.weights is not None:
    for name, weight in zip(feature_names, model.weights.flatten()):
        print(f"{name}: {weight:.4f}")  