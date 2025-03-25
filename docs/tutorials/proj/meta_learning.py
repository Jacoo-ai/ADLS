# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load experiment result data
df = pd.read_csv("docs/tutorials/proj/results/A100/resnet18_experiment_results.csv")

# Normalize numerical features: latency, accuracy, energy
scaler = MinMaxScaler()
df[["latency", "accuracy", "energy"]] = scaler.fit_transform(df[["latency", "accuracy", "energy"]])

# Encode categorical labels to integer indices
model_le = LabelEncoder()
batch_le = LabelEncoder()
quant_le = LabelEncoder()


df["model_label"] = model_le.fit_transform(df["model_name"])
df["batch_label"] = batch_le.fit_transform(df["batch_size"])
df["quant_label"] = quant_le.fit_transform(df["quant_method"])


# Extract input features and target labels
X = df[["latency", "accuracy", "energy"]].values
y_model = df["model_label"].values
y_batch = df["batch_label"].values
y_quant = df["quant_label"].values

# Split the dataset into training and testing sets
X_train, X_test, y_model_train, y_model_test, y_batch_train, y_batch_test, y_quant_train, y_quant_test = train_test_split(
    X, y_model, y_batch, y_quant, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_model_tensor = torch.LongTensor(y_model_train)
y_batch_tensor = torch.LongTensor(y_batch_train)
y_quant_tensor = torch.LongTensor(y_quant_train)

X_test_tensor = torch.FloatTensor(X_test)

# Define the multi-head neural network for multi-task classification
class MultiHeadPolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, model_classes, batch_classes, quant_classes):
        super(MultiHeadPolicyNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.model_head = nn.Linear(hidden_dim, model_classes)
        self.batch_head = nn.Linear(hidden_dim, batch_classes)
        self.quant_head = nn.Linear(hidden_dim, quant_classes)

    def forward(self, x):
        shared = self.shared(x)
        return self.model_head(shared), self.batch_head(shared), self.quant_head(shared)

# Set model parameters
input_dim = 3
hidden_dim = 64
model_classes = len(model_le.classes_)
batch_classes = len(batch_le.classes_)
quant_classes = len(quant_le.classes_)

# Instantiate the model, loss function, and optimizer
model = MultiHeadPolicyNetwork(input_dim, hidden_dim, model_classes, batch_classes, quant_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    pred_model, pred_batch, pred_quant = model(X_train_tensor)

    loss_model = criterion(pred_model, y_model_tensor)
    loss_batch = criterion(pred_batch, y_batch_tensor)
    loss_quant = criterion(pred_quant, y_quant_tensor)

    total_loss = loss_model + 10 * loss_batch + loss_quant
    total_loss.backward()
    optimizer.step()

    # Log progress every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Total Loss={total_loss.item():.4f} | model={loss_model.item():.4f}, batch={loss_batch.item():.4f}, quant={loss_quant.item():.4f}")

print("Training completed.")

# Inference and decoding predictions
model.eval()
with torch.no_grad():
    pred_model, pred_batch, pred_quant = model(X_test_tensor)

    pred_model_labels = torch.argmax(pred_model, dim=1).numpy()
    pred_batch_labels = torch.argmax(pred_batch, dim=1).numpy()
    pred_quant_labels = torch.argmax(pred_quant, dim=1).numpy()

    # Decode predicted labels to original category names
    model_pred_names = model_le.inverse_transform(pred_model_labels)
    batch_pred_sizes = batch_le.inverse_transform(pred_batch_labels)
    quant_pred_methods = quant_le.inverse_transform(pred_quant_labels)

    # Display a few prediction examples
    print("\nSample Predictions:")
    for i in range(5):
        print(f"Input: {X_test[i]}")
        print(f"    Predicted: {model_pred_names[i]}, batch={batch_pred_sizes[i]}, quant={quant_pred_methods[i]}")
        print(f"    True: {model_le.inverse_transform([y_model_test[i]])[0]}, batch={batch_le.inverse_transform([y_batch_test[i]])[0]}, quant={quant_le.inverse_transform([y_quant_test[i]])[0]}")
        print("")

# Configuration recommendation function
def recommend_best_config(*, target_latency, target_accuracy, target_energy):
    """
    Recommend the most suitable model_name, batch_size, and quant_method
    based on target latency, accuracy, and energy constraints.
    """
    # Prepare input DataFrame for normalization
    input_df = pd.DataFrame([{
        "latency": target_latency,
        "accuracy": target_accuracy,
        "energy": target_energy
    }])

    # Normalize using previously fitted scaler
    scaled_input = scaler.transform(input_df)

    # Convert to tensor for model inference
    input_tensor = torch.FloatTensor(scaled_input)

    # Perform forward pass
    model.eval()
    with torch.no_grad():
        pred_model, pred_batch, pred_quant = model(input_tensor)

    # Decode prediction to original labels
    model_idx = torch.argmax(pred_model, dim=1).item()
    batch_idx = torch.argmax(pred_batch, dim=1).item()
    quant_idx = torch.argmax(pred_quant, dim=1).item()

    best_model = model_le.inverse_transform([model_idx])[0]
    best_batch_size = batch_le.inverse_transform([batch_idx])[0]
    best_quant_method = quant_le.inverse_transform([quant_idx])[0]

    return {
        "Model": best_model,
        "Batch Size": best_batch_size,
        "Quant Method": best_quant_method
    }

# Example usage
recommendation = recommend_best_config(
    target_latency=0.5,
    target_accuracy=0.7,
    target_energy=0.002
)

print("Recommended Result:", recommendation)

def decode_prediction(pred_tuple):
    """Decode prediction tensors from the model's 3-head outputs."""
    pred_model_tensor, pred_batch_tensor, pred_quant_tensor = pred_tuple

    model_preds = pred_model_tensor.argmax(dim=1).cpu().numpy()
    batch_preds = pred_batch_tensor.argmax(dim=1).cpu().numpy()
    quant_preds = pred_quant_tensor.argmax(dim=1).cpu().numpy()

    return model_preds, batch_preds, quant_preds


def evaluate_model(model, X_test_tensor, y_model_test, y_batch_test, y_quant_test):
    """
    Evaluate the model and report accuracy for each prediction head,
    as well as full-match accuracy across all three outputs.
    """
    model.eval()
    with torch.no_grad():
        pred_tuple = model(X_test_tensor)

    pred_model, pred_batch, pred_quant = decode_prediction(pred_tuple)

    acc_model = np.mean(pred_model == y_model_test)
    acc_batch = np.mean(pred_batch == y_batch_test)
    acc_quant = np.mean(pred_quant == y_quant_test)

    acc_full = np.mean(
        (pred_model == y_model_test) &
        (pred_batch == y_batch_test) &
        (pred_quant == y_quant_test)
    )

    print(f"Model Accuracy:        {acc_model:.4f}")
    print(f"Batch Size Accuracy:   {acc_batch:.4f}")
    print(f"Quant Method Accuracy: {acc_quant:.4f}")
    print(f"Full Match Accuracy:   {acc_full:.4f}")

    return {
        "model_acc": acc_model,
        "batch_acc": acc_batch,
        "quant_acc": acc_quant,
        "full_match_acc": acc_full
    }

# Call the evaluation function
metrics = evaluate_model(model, X_test_tensor, y_model_test, y_batch_test, y_quant_test)
