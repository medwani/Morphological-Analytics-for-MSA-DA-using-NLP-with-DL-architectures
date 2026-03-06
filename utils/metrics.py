
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

def calculate_wla(y_true, y_pred):
    """Calculates Word-Level Accuracy (WLA)."""
    # Assumes y_true and y_pred are lists of lists, where each inner list is a sequence of tags for a word.
    correct_words = 0
    total_words = len(y_true)
    for true_seq, pred_seq in zip(y_true, y_pred):
        if all(t == p for t, p in zip(true_seq, pred_seq)):
            correct_words += 1
    return correct_words / total_words if total_words > 0 else 0

def calculate_fla(y_true, y_pred):
    """Calculates Feature-Level Accuracy (FLA)."""
    # Flatten the lists of lists into a single list of tags
    true_flat = [item for sublist in y_true for item in sublist]
    pred_flat = [item for sublist in y_pred for item in sublist]
    return accuracy_score(true_flat, pred_flat)

def calculate_lemmatization_accuracy(true_lemmas, pred_lemmas):
    """Calculates Lemmatization Accuracy (LA)."""
    return accuracy_score(true_lemmas, pred_lemmas)

def calculate_segmentation_accuracy(true_segments, pred_segments):
    """Calculates Segmentation Accuracy (SA)."""
    return accuracy_score(true_segments, pred_segments)

def calculate_ambiguity_resolution_accuracy(y_true_ambiguous, y_pred_ambiguous):
    """Calculates Ambiguity Resolution Accuracy (ARA)."""
    return accuracy_score(y_true_ambiguous, y_pred_ambiguous)

def calculate_dtg(native_accuracy, transfer_accuracy):
    """Calculates Dialect Transfer Gap (DTG)."""
    if native_accuracy == 0:
        return 0
    return (native_accuracy - transfer_accuracy) / native_accuracy

def calculate_drs(accuracies):
    """Calculates Dialect Robustness Score (DRS)."""
    if not accuracies:
        return 0
    # Harmonic mean of accuracies
    return len(accuracies) / sum(1.0 / acc for acc in accuracies if acc > 0)

def calculate_csp(y_true_cs, y_pred_cs):
    """Calculates Code-Switching Performance (CSP)."""
    return accuracy_score(y_true_cs, y_pred_cs)

def get_all_metrics(results):
    """A placeholder function to calculate and format all metrics for tables."""
    # In a real implementation, this would take detailed prediction outputs
    # and generate the tables seen in the dissertation.
    print("Calculating all metrics...")
    # This would call the functions above and format the output.
    metrics = {
        "WLA": 0.943, # Placeholder from dissertation Table 12
        "FLA_POS": 0.981, # Placeholder from dissertation Table 13
        "DTG_EGY_to_LEV": 0.05, # Placeholder from dissertation Table 17
        "DRS": 0.84 # Placeholder
    }
    return metrics
