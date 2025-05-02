# TASK-6-K-Nearest-Neighbors-KNN-Classification
 Euclidean distance &amp; K selection- Iris Dataset

 # Step-by-Step Breakdown
Training Phase (knn.fit(X_train, y_train)):

For KNN, this step doesn't actually "train" the model in the traditional sense.

It stores the training data, because KNN is a lazy learner—it only learns when it needs to make a prediction.

Prediction Phase (knn.predict(X_test)):

For each point in X_test, KNN:

Calculates the distance from this point to every point in X_train (typically using Euclidean distance).

Selects the k=3 nearest neighbors (as you set n_neighbors=3).

Looks at their labels (from y_train).

Performs a majority vote to decide the class of the test point.

# Example (Simplified):
Suppose you're trying to classify a new flower sample:

PetalLengthCm = 5.1

PetalWidthCm = 1.8

KNN will:

Measure distances between this sample and all training samples.

Pick the 3 closest ones (smallest distances).

Check which species those 3 neighbors belong to.

Assign the most frequent species label among those 3 neighbors to the new sample.

# Why it Works Well Here:
The Iris dataset is small and fairly well-separated (especially with all 4 features), which makes KNN effective.

It’s a non-parametric model, meaning it doesn’t assume any distribution for the data.

# Steps to Choose the Right k in Your Code:
1. Try multiple values of k
2. Interpret the Plot
3. Use Cross-Validation (Optional for robustness)

# How KNN Works Recap
KNN classifies a new point by:

Measuring distance (usually Euclidean distance) to all training points.

Distance formula:

 distance=(x1-x2)2+(y1-y2)2+...

​# Problem Without Normalization:
In your dataset, feature scales are very different:

PetalLengthCm might range from 1 to 6.

SepalWidthCm might range from 2 to 4.5.

This means:

Larger-scaled features (like PetalLengthCm) will dominate the distance calculations.

The model will bias toward those features, even if others are more relevant for classification.

# Why Normalization Fixes This:
Normalization rescales all features to a similar range—commonly:

Min-Max scaling (range [0, 1])

Standardization (mean 0, std 1)

This ensures all features contribute equally to distance calculations.

# 1. Training Time Complexity
KNN has no actual training phase, since it's a lazy learner. It just stores the training data.

Training Time Complexity:

𝑂(1)
O(1)
KNN doesn't build a model or perform computations during training.

# 2. Prediction Time Complexity
When making a prediction for each test sample, KNN:

Calculates distances from the test sample to all n training samples.

Sorts or partially sorts distances to find the k nearest.

Picks the majority class from the k closest.

For each prediction:
Distance calculation: O(n × d), where:

n = number of training samples

d = number of features (4 in your case)

Finding the k smallest distances: O(n log k) or O(n) with a heap

For m test samples:
Total prediction time:
O(m×n×d)
In your case:

n ≈ 105 (70% of 150 samples)

m ≈ 45 (30% of 150 samples)

d = 4 features

So while this is fast for small datasets like Iris, KNN scales poorly for large datasets due to the linear dependency on the number of training samples.


# Pros of KNN in Your Code
1. Simple and Intuitive
KNN is very easy to understand and implement.

No complex math or assumptions needed—just distance comparisons.

2. No Training Required
Since it’s a lazy learner, there's no training cost.

Useful for quick prototyping.

3. Adapts Well to Small Datasets
The Iris dataset is small (150 samples), so KNN performs well.

Memory and compute requirements are manageable.

4. Non-Parametric
KNN doesn’t assume a specific form of the decision boundary (e.g., linear).

It can capture complex class shapes if enough data is available.

# Cons of KNN in Your Code
1. Sensitive to Feature Scale
Without normalization (e.g., using StandardScaler), features like PetalLengthCm dominate distance calculations.

Fix: Normalize features before applying KNN.

2. Prediction is Slow
Time complexity is O(m × n × d)—inefficient for large datasets.

For each test point, it computes distances to all training samples.

3. Curse of Dimensionality
With more features (dimensions), distance measures become less meaningful.

Fortunately, you’re only using 4 features, so this is not a big issue here.

4. Sensitive to Noise and Outliers
Noisy or misclassified points in training data can distort predictions.

Can be improved using larger k values or weighting neighbors by distance.

5. Memory-Intensive
The model must store the entire training set, which can be costly in large-scale applications.

# In the Context of the Iris Dataset:
KNN performs well here because:

The dataset is clean and small.

The classes are well-separated (especially in petal features).

But KNN would struggle on large, noisy, or high-dimensional datasets unless optimized.


# Why KNN is Sensitive to Noise
KNN makes predictions based on the labels of the nearest neighbors. If any of those neighbors are:

Labeled incorrectly, or

Outliers that don’t represent the typical distribution of a class,

...then the KNN algorithm might be misled, leading to incorrect predictions.

# Example Using Your Code
Suppose you accidentally have a mislabeled sample:

plaintext
Copy
Edit
Sample with petal length = 5.1, petal width = 1.8
Labelled as Iris-setosa (should be Iris-versicolor)
Now, during prediction:

If this mislabeled sample ends up as one of the 3 nearest neighbors, it could skew the majority vote.

This can cause a correctly classified test point to be misclassified.

# How It Impacts Your KNN Results
In small datasets like Iris (150 samples), even a few noisy points can noticeably affect accuracy.

Since KNN doesn't "learn" patterns, it can't distinguish between "typical" and "atypical" samples.

Logistic Regression or decision trees are less affected by noise, since they learn generalized rules.

🛠# Mitigating Noise Sensitivity
Here are practical ways to reduce the impact in your code:

Increase k

A higher k (like 5 or 7) reduces the influence of any single noisy neighbor.

Use Distance Weighting

Assign higher importance to closer neighbors:

python
Copy
Edit
KNeighborsClassifier(n_neighbors=3, weights='distance')
Clean the Data (if possible)

Remove or fix outliers or mislabeled points.

Use Cross-Validation

Helps detect if your model is consistently affected by outliers across different splits.


# How KNN Handles Multi-Class Problems
KNN does not require special modifications for multi-class classification.

# Here's how it works in your case:
For each test sample, KNN:

Calculates distances to all training samples.

Finds the k nearest neighbors.

Checks the class labels of those neighbors.

Performs a majority vote.

Since class labels are:

Encoded numerically (0, 1, 2) using LabelEncoder,

KNN just counts how many neighbors belong to each class and assigns the class with the most votes.

# Example (with k=5):
Suppose a test point has 5 nearest neighbors with these class labels:

[1, 1, 2, 2, 2]

Then:

Class 2 appears 3 times → predicted label is 2 (which might map to virginica).

# Important Notes:
✔# Scikit-learn's KNeighborsClassifier handles this internally:
python
Copy
Edit
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
No need for One-vs-One or One-vs-Rest strategies.

Works directly on integer-labeled classes.

# In the Iris Dataset:
All 3 species are well-distributed and separable in feature space (especially petal length/width).

KNN performs well on this multi-class task, especially when features are normalized.


# Role of Distance Metrics in KNN
# Core Idea:
KNN classifies a new point based on the k closest training samples using a distance metric.

The choice of distance metric directly affects:

Which neighbors are considered “nearest”

The final predicted class

# Default in Your Code: Euclidean Distance
In scikit-learn, the default metric is Euclidean distance:
 distance=(x1-x2)2+(y1-y2)2+...

​This works well when all features are normalized, which is why scaling is important (as we discussed earlier).

# Other Common Distance Metrics:
Metric	Formula	When to Use
Manhattan (L1)	(\sum	x_i - y_i
Minkowski	Generalized version of Euclidean and Manhattan (parameter p)	You can tune p for flexibility
Cosine	
1− ∥x∥∥y∥/x⋅y
​
 	Useful for text data or high-dimensional spaces

