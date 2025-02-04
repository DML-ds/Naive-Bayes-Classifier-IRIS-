import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=60)

# Calculate mean and standard deviation for each feature and class
num_classes = len(np.unique(y))
num_features = X.shape[1]

mean_features = np.zeros((num_classes, num_features))
std_features = np.zeros((num_classes, num_features))

for class_idx in range(num_classes):
    X_class = X_train[y_train == class_idx]
    mean_features[class_idx] = np.mean(X_class, axis=0)
    std_features[class_idx] = np.std(X_class, axis=0)

# Gaussian likelihood function
def gaussian_likelihood(x, mean, std):
    numerator = np.exp(-0.5 * ((x - mean) / std) ** 2)
    denominator = np.sqrt(2 * np.pi) * std
    return numerator / denominator

# Calculate prior probabilities
prior_probabilities = np.bincount(y_train) / len(y_train)

# Calculate posterior probabilities for a sample
def posterior_probability(sample):
    posteriors = []
    for class_idx in range(num_classes):
        likelihoods = np.prod(gaussian_likelihood(sample, mean_features[class_idx], std_features[class_idx]))
        posterior = likelihoods * prior_probabilities[class_idx]
        posteriors.append(posterior)
    return posteriors

# Predict class labels for the test set
y_pred = np.argmax([posterior_probability(sample) for sample in X_test], axis=1)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Prediction accuracy on test set: {accuracy:.4f}")

# Generate new samples based on learned distributions
def generate_sample(class_idx):
    return np.random.normal(mean_features[class_idx], std_features[class_idx])

num_samples = 10
new_samples = np.array([generate_sample(class_idx) for _ in range(num_samples) for class_idx in range(num_classes)])
new_labels = np.array([class_idx for _ in range(num_samples) for class_idx in range(num_classes)])

# Print generated samples and their labels
print("\nGenerated Samples:")
for i in range(num_samples * num_classes):
    class_name = iris.target_names[new_labels[i]]
    print(f"Sample {i+1} (Class: {class_name}):", new_samples[i])
