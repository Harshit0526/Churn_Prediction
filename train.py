from churn_analysis.modeling import train_and_save


if __name__ == "__main__":
    bundle = train_and_save()
    print(f"Best model: {bundle['model_name']}")
    print(f"ROC-AUC: {bundle['metrics'][bundle['model_name']]['roc_auc']:.3f}")
