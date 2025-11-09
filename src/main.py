import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.components.data_preprocessing import preprocess_and_upload
from src.components.make_dataset import make_dataset
from src.components.model_trainer import train
from src.components.model_predictor import predict

logger = get_logger(__name__)
load_dotenv()

def main():
    logger.info("Starting full pipeline")
    raw = Path("data/raw/sales_original.csv")
    cleaned = Path("data/processed/sales_clean.csv")
    interim = Path("data/interim")
    # 1. Preprocess and upload to Postgres
    preprocess_and_upload(str(raw), str(cleaned), upload_to_db=True)
    # 2. Make dataset
    make_dataset(str(cleaned), str(interim), test_ratio=float(os.getenv("TEST_RATIO", 0.2)))
    # 3. Train (MLflow URI via env)
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_path, rmse = train(str(interim / "train.csv"), str(interim / "test.csv"), "params.yaml", mlflow_uri)
    # 4. Predict
    predict(model_path, str(interim / "test.csv"), "data/processed/predictions.csv")
    # 5. DVC add & push suggested (not run automatically)
    logger.info("Pipeline finished. Consider running: dvc add data/processed/sales_clean.csv models/prophet_model.pkl && dvc push")
    logger.info("Done.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.components.data_preprocessing import preprocess_and_upload
from src.components.make_dataset import make_dataset
from src.components.model_trainer import train
from src.components.model_predictor import predict

logger = get_logger(__name__)
load_dotenv()

def main():
    logger.info("Starting full pipeline")
    raw = Path("data/raw/sales_original.csv")
    cleaned = Path("data/processed/sales_clean.csv")
    interim = Path("data/interim")
    # 1. Preprocess and upload to Postgres
    preprocess_and_upload(str(raw), str(cleaned), upload_to_db=True)
    # 2. Make dataset
    make_dataset(str(cleaned), str(interim), test_ratio=float(os.getenv("TEST_RATIO", 0.2)))
    # 3. Train (MLflow URI via env)
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_path, rmse = train(str(interim / "train.csv"), str(interim / "test.csv"), "params.yaml", mlflow_uri)
    # 4. Predict
    predict(model_path, str(interim / "test.csv"), "data/processed/predictions.csv")
    # 5. DVC add & push suggested (not run automatically)
    logger.info("Pipeline finished. Consider running: dvc add data/processed/sales_clean.csv models/prophet_model.pkl && dvc push")
    logger.info("Done.")

if __name__ == "__main__":
    main()
