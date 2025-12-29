"""Comprehensive evaluation metrics and visualization for model performance."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with metrics and visualizations."""
    
    def __init__(self, class_names: List[str] | None = None):
        """Initialize evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names or ["NORMAL", "PNEUMONIA"]
        self.metrics: Dict = {}
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels (binary: 0/1)
            y_pred: Predicted labels (binary: 0/1)
            y_pred_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            "accuracy": float(np.mean(y_true == y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        
        # ROC-AUC (only if we have both classes)
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
        else:
            logger.warning("Only one class present in y_true. Skipping ROC-AUC calculation.")
            metrics["roc_auc"] = None
        
        # Calculate specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)
        
        self.metrics = metrics
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[str | Path] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str | Path] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            save_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_path: Optional[str | Path] = None,
        figsize: Tuple[int, int] = (8, 6),
    ) -> plt.Figure:
        """Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities for positive class
            save_path: Optional path to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        return fig
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        output_dir: str | Path,
        model_name: str = "model",
    ) -> Dict[str, any]:
        """Generate comprehensive evaluation report with all metrics and plots.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            output_dir: Directory to save plots
            model_name: Name of the model for file naming
            
        Returns:
            Dictionary containing all metrics and plot paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Generate plots
        plots = {}
        
        # Confusion Matrix
        cm_path = output_dir / f"{model_name}_confusion_matrix.png"
        self.plot_confusion_matrix(y_true, y_pred, save_path=cm_path)
        plots["confusion_matrix"] = str(cm_path)
        plt.close()
        
        # ROC Curve
        if metrics.get("roc_auc") is not None:
            roc_path = output_dir / f"{model_name}_roc_curve.png"
            self.plot_roc_curve(y_true, y_pred_proba, save_path=roc_path)
            plots["roc_curve"] = str(roc_path)
            plt.close()
        
        # Precision-Recall Curve
        pr_path = output_dir / f"{model_name}_precision_recall_curve.png"
        self.plot_precision_recall_curve(y_true, y_pred_proba, save_path=pr_path)
        plots["precision_recall_curve"] = str(pr_path)
        plt.close()
        
        logger.info(f"Evaluation report generated for {model_name}")
        logger.info(f"Metrics: {metrics}")
        
        return {
            "metrics": metrics,
            "plots": plots,
        }
    
    def print_metrics_summary(self) -> None:
        """Print a formatted summary of metrics."""
        if not self.metrics:
            logger.warning("No metrics calculated yet.")
            return
        
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        
        for metric, value in self.metrics.items():
            if isinstance(value, float):
                print(f"{metric.replace('_', ' ').title():.<30} {value:.4f}")
            else:
                print(f"{metric.replace('_', ' ').title():.<30} {value}")
        
        print("="*50 + "\n")
