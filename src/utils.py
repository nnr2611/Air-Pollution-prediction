import os

def ensure_directory(directory):
    """Ensure a directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_results(results, file_path):
    """Save results to a text file."""
    with open(file_path, 'w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")

def plot_results(true_values, predictions, title="Model Predictions"):
    """Plot true vs. predicted values."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label="True Values", marker="o")
    plt.plot(predictions, label="Predictions", marker="x")
    plt.title(title)
    plt.legend()
    plt.show()
