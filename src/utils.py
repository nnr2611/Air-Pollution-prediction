import json
from pathlib import Path


def ensure_directory(directory):
    """Create ``directory`` (and parents) if it does not exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_results(results, file_path):
    """Save a dict of results as JSON."""
    ensure_directory(Path(file_path).parent)
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)


def plot_results(true_values, predictions, title="Model Predictions", path=None):
    """Plot true vs. predicted values; save to ``path`` if given, else show."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label="Actual")
    plt.plot(predictions, label="Predicted", alpha=0.8)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=120)
        plt.close()
    else:
        plt.show()
