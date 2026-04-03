from src.eda.exploratory_data_analysis import run_eda
from src.preprocessing.preprocessing import run_preprocessing
from src.tuning import (
    tune_knn,
    tune_logistic_regression,
    tune_random_forest,
)
from src.modeling import (
    evaluate_knn,
    evaluate_logistic_regression,
    evaluate_random_forest
)
from src.evaluation.evaluation import run_evaluation
from src.explainability import run_importance, run_shap

def print_metrics(result: dict) -> None:
    print(f"\n=== {result['model_name'].upper()} ===")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-Score: {result['f1_score']:.4f}")
    print("\nClassification Report:")
    print(result["classification_report"])

def print_tuning_result(model_name: str, best_params: dict, best_score: float) -> None:
    print(f"\n=== TUNING - {model_name.upper()} ===")
    print(f"Best Params: {best_params}")
    print(f"Best CV Recall: {best_score:.4f}")

def main():
    df = run_eda()
    data = run_preprocessing(df)

    x_train, x_test, y_train, y_test, x_train_scaled, x_test_scaled = (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["y_test"],
        data["x_train_scaled"],
        data["x_test_scaled"]
    )

    knn_model, knn_best_params, knn_best_score = tune_knn(x_train_scaled, y_train)
    print_tuning_result("KNN", knn_best_params, knn_best_score)
    knn_result = evaluate_knn(knn_model, x_test_scaled, y_test)
    print_metrics(knn_result)

    logistic_model, logistic_best_params, logistic_best_score = tune_logistic_regression(x_train_scaled, y_train)
    print_tuning_result("Logistic Regression", logistic_best_params, logistic_best_score)
    logistic_result = evaluate_logistic_regression(logistic_model, x_test_scaled, y_test)
    print_metrics(logistic_result)

    random_forest_model, rf_best_params, rf_best_score = tune_random_forest(x_train, y_train)
    print_tuning_result("Random Forest", rf_best_params, rf_best_score)
    random_forest_result = evaluate_random_forest(random_forest_model, x_test, y_test)
    print_metrics(random_forest_result)

    results = [knn_result, logistic_result, random_forest_result]
    run_evaluation(results, y_test)

    run_importance(
        model=random_forest_model,
        feature_names=list(x_train.columns),
        top_n=10
    )

    run_shap(
        model=random_forest_model,
        x_data=x_train,
        feature_names=list(x_train.columns)
    )

    print("FIM")

if __name__ == '__main__':
    main()