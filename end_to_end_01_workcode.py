import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading ---
con = duckdb.connect(database=":memory:") # in-memory database, no need to persist — data is rebuilt from CSVs each run

con.execute("""
    CREATE VIEW orders AS SELECT * FROM 'data/olist_orders_dataset.csv';
    CREATE VIEW payments AS SELECT * FROM 'data/olist_order_payments_dataset.csv';
    CREATE VIEW reviews AS SELECT * FROM 'data/olist_order_reviews_dataset.csv';
""")

res_total = con.execute("SELECT count(*) FROM orders").df()
print(res_total) # total number of inputs

res_dist = con.execute("""
    SELECT 
        review_score, 
        count(*) as count 
    FROM reviews 
    GROUP BY review_score 
    ORDER BY review_score
""").df()
print(res_dist) # distribution of review_score

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

# --- Model Definition --
class LogisticRegression():
    def __init__(self):
        # with higher learning rate model was jumping out of a local maximum
        # this learning rate pushes model into polishing and converging with final epochs
        self.learning_rate = 0.005 
        self.epochs = 15000 # convergence around 10k, extra margin for stability
        self.weights = None # starting with None, updating throughout looping phase

    def sigmoid(self, z):
        z_clipped = np.clip(z, -250, 250) # avoiding runtime warning due to too low/high float
        return 1 / (1 + np.exp(-z_clipped))
    
    def compute_log_likelihood(self, X, y):
        z = X @ self.weights # dot product of feature matrix and weights
        p = self.sigmoid(z) # squishing it into sigmoid to calculate probability <0, 1>

        epsilon = 1e-15 # preventing np.log(0) to avoid infinity

        # measures how well predicted probability match actual labels
        # we are punishing bad prediction with higher penalties through logarithms
        log_likelihood = y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon)

        return np.sum(log_likelihood) # sum of log-likelihood across rows (customers)
    
    

    def fit(self, X, y):
        n_columns = X.shape[1] # shape is (rows, columns)
        self.weights = np.zeros((n_columns, 1)) # zeros because we are starting with zero knowledge

        n_rows = X.shape[0]
        # as we have more positive than negative reviews we are giving negative reviews more weight
        # good review is counted as 1 and bad as 2.5 (hyperparameter) 
        weights_vector = np.where(y == 1, 2.5, 1) 
        for i in range(self.epochs):
            z = X @ self.weights # multiplying feature matrix values by weights and summing them together
            p = self.sigmoid(z) 

            # (1 / n_rows) gives us average of the gradient
            # (y - p) gives us error, how far we are with our prediction from reality
                # * weights_vector using it for punishing bad reviews
            # X.T because original matrix is (96000, 4)
                # in order to get a dot product with weights vector (96000, 1) we need to transpone it
            gradient = (1 / n_rows) * (X.T @ ((y - p) * weights_vector))

            # updating self.weights by gradient multiplied by LR (hyperparameter)
            self.weights += self.learning_rate * gradient

            if i % 1000 == 0:
                # printing current LL every 1000 epoch
                current_LL = self.compute_log_likelihood(X, y) 

                print(f"Current epoch: {i}, with error: {current_LL}")

    def predict_probab(self, X):
        # returining updated feature matrix by weights squished into sigmoid
        return self.sigmoid(X @ self.weights)
    
    def predict(self, X, threshold=0.375):
        probs = self.predict_probab(X) # probability of our updated matrix 
        return (probs >= threshold).astype(int) # comparing to set threshold (hyperparameter)

# --- Fitting Model ---        
model = LogisticRegression()
# extracting column is_bad_review containing 0 or 1
# converting it to np.array
# reshaping it to 2D with one column of these values (column vector)
vector_y = final_df["is_bad_review"].to_numpy().reshape(-1, 1).astype(float)
# extracting features which we want for weights
features = ["max_installments","sum_value", "delivery_delay"]
# standardization (z-score normalization) in order to prevent domination by some weights
# squishing them into normal distribution
X_scaled = (final_df[features] - final_df[features].mean()) / final_df[features].std()
X_array = X_scaled.to_numpy()
# creates column with 1 of n_rows 
ones_array = np.ones((X_array.shape[0], 1))
# stacks 2 arrays side-by-side horizontally
# so X expands from 3 columns to 4 with first column being the column with 1
X_array = np.hstack([ones_array, X_array]).astype(float)

# final model takes X_array with 4 columns 
# and column vector_y with 0 or 1 indicating bad review 
model.fit(X_array, vector_y)

predictions = model.predict(X_array) # comparison to threshold

# --- Model Evaluation ---
# confusion matrix
tp = np.sum((predictions == 1) & (vector_y == 1)) # reality: bad review, predicted: bad review
tn = np.sum((predictions == 0) & (vector_y == 0)) # reality: good review, predicted: good reviewv
fp = np.sum((predictions == 1) & (vector_y == 0)) # reality: good review, predicted: bad review
fn = np.sum((predictions == 0) & (vector_y == 1)) # reality: bad review, predicted: good review

accuracy = (tp + tn) / len(vector_y)
precision = tp / (tp + fp + 1e-15)
recall = tp / (tp + fn + 1e-15)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)

print(f"Model precision is {precision:.2%}")
print(f"Model accuracy is: {accuracy:.2%}")
print(f"Model recall is {recall:.2%}")
print(f"Model F1 score is {f1_score:.2%}")

# setting up names for different outputs with first being the intercept and other being weights
feature_names = ["Intercept", "Installments","Price", "Delay"]
# default of model.weights is None
if model.weights is not None:
    # zipping feature names and weights together, naming every output of weights
    # model.weights.flatten() transforms it from a column vector into 1D vector (like number line)
    for name, weight in zip(feature_names, model.weights.flatten()):
        print(f"{name}: {weight:.4f}")

# --- Results Visualisation ---
probs = model.predict_probab(X_array) # one prediction with which will visualisation operate

from matplotlib.widgets import Slider

fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # one figure with 3 charts side-by-side; 18, 6 is figsize in inches
fig.subplots_adjust(bottom=0.2) # shrinks plot upward to make room for the slider at the bottom

# weight bar chart (static)
if model.weights is not None:
    weights_flat = model.weights.flatten()
    feature_names_plot = ["Installments", "Price", "Delay"]
    feature_weights = weights_flat[1:] # selects only weights not intercept
    colors_bar = ['#AEC6CF', '#AEC6CF', '#1F3864']
    axes[0].barh(feature_names_plot, feature_weights, color=colors_bar) # horizontal bar chart
    axes[0].set_xlabel("Weight")
    axes[0].set_title("Impact of weights on a bad review")
    axes[0].spines["top"].set_visible(False) # making top borderline invisible
    axes[0].spines["right"].set_visible(False) # making right borderline invisible
    axes[0].xaxis.grid(True, linestyle="--", alpha=0.5) # reference points at regular intervals
    axes[0].set_axisbelow(True) # gridlines are set behind the bars

# slider
# reserves rectangle for a slider
# 35% from the left, 6% from the bottom, 30% wide and 3% tall
slider_ax = fig.add_axes((0.35, 0.06, 0.3, 0.03))
# puts slider in rectangle, label Threshold, 0.1 min, 0.9 max, 0.375 is the starting position
threshold_slider = Slider(slider_ax, 'Threshold', 0.1, 0.9, valinit=0.375)

def update(val):
    t = threshold_slider.val # grabs the current position of the slider 
    preds = (probs >= t).astype(int) # compares customer probability to the threshold and set to 0 or 1

    tp = np.sum((preds == 1) & (vector_y == 1))
    tn = np.sum((preds == 0) & (vector_y == 0))
    fp = np.sum((preds == 1) & (vector_y == 0))
    fn = np.sum((preds == 0) & (vector_y == 1))

    precision = tp / (tp + fp + 1e-15)
    recall = tp / (tp + fn + 1e-15)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-15)
    accuracy = np.mean(preds == vector_y)

    # confusion matrix
    axes[1].clear()
    cm = np.array([[tn, fp], [fn, tp]])
    row0 = tn + fp
    row1 = fn + tp
    annot = np.array([[f"TN: {tn}\n({tn/row0:.2%})", f"FP: {fp}\n({fp/row0:.2%})"],
                      [f"FN: {fn}\n({fn/row1:.2%})", f"TP: {tp}\n({tp/row1:.2%})"]])
    norm_cm = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(norm_cm, annot=annot, fmt="", ax=axes[1], cbar=False)
    axes[1].set_title(f"Confusion Matrix (threshold={t:.2f})")

    # metrics
    axes[2].clear()
    metrics = [accuracy, precision, recall, f1]
    names = ["Accuracy", "Precision", "Recall", "F1"]
    colors_met = ["#00FF3C", '#AEC6CF', '#AEC6CF', '#1F3864']
    axes[2].barh(names, metrics, color=colors_met)
    axes[2].set_xlim(0, 1) # fixes x-axis between 0, 1
    axes[2].set_title("Metrics")
    # for loop for naming the bars
    # v + 0.02 is the horizontal position
    # i is the vertical position
    # f"{v:.2%}" puts there the percentage of the bar
    for i, v in enumerate(metrics):
        axes[2].text(v + 0.02, i, f"{v:.2%}", va='center')

    fig.canvas.draw_idle() # pushes new visual into the window

update(None) # draw everything with default threshold
threshold_slider.on_changed(update) # connects slider to the update function
plt.show()