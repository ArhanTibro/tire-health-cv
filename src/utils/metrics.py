"""
Shared evaluation utilities so CNN and YOLO results are directly comparable.

Should produce, per model run, saved to results/<model_name>/:
  - per-class precision / recall / F1
  - confusion matrix (plot + raw values)
  - overall accuracy

TODO (next phase): implement using sklearn.metrics (classification_report,
confusion_matrix) + matplotlib/seaborn for the plot.
"""
