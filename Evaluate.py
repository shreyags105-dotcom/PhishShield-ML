import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def evaluate_model(model_name, y_test, y_pred):
    """Calculates and prints core metrics for a single model."""
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"=== Performance Report: {model_name} ===")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}\n")
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 40)
    
    return {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Generates and displays a confusion matrix heatmap."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    plt.show()

def plot_model_comparison(model_scores):
    """
    Takes a dictionary like: 
    {'Decision Tree': 0.92, 'Random Forest': 0.96, 'Logistic Regression': 0.88}
    and plots a comparative bar graph.
    """
    models = list(model_scores.keys())
    accuracies = list(model_scores.values())
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy Score')
    plt.title('PhishShield Model Accuracy Comparison')
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')
        
    plt.tight_layout()
    plt.show()

# --- Example of how you will use this once Riya hands over predictions ---
if __name__ == "__main__":
    # Dummy data test (Replace y_test and y_pred with Riya's actual test arrays later)
    y_true_example = [0, 1, 1, 0, 1, 0, 0, 1]
    y_pred_example = [0, 1, 0, 0, 1, 0, 1, 1]
    
    # 1. Test metrics function
    evaluate_model("Random Forest (Demo)", y_true_example, y_pred_example)
    
    # 2. Test confusion matrix plot
    plot_confusion_matrix(y_true_example, y_pred_example, "Random Forest")
    
    # 3. Test comparison chart
    sample_comparison = {
        'Decision Tree': 0.91,
        'Random Forest': 0.95,
        'Logistic Regression': 0.85
    }
    plot_model_comparison(sample_comparison)