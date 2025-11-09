mkdir -p mlflow_tracking/mlflow_artifacts
mlflow server \
  --backend-store-uri sqlite:///mlflow_tracking/mlflow.db \
  --default-artifact-root file:$(pwd)/mlflow_tracking/mlflow_artifacts \
  --host 0.0.0.0 -p 5000
