# 🧠 Sales Demand Forecasting using Prophet & MLflow

**Author:** [Bishal Nengminja](https://github.com/Bishal-Nengminja)  
**Tech Stack:** Python, Prophet, MLflow, DVC, Pandas, Matplotlib, VS Code  

---

## 📊 Project Overview
This project predicts **future sales demand** based on historical transaction data.  
It uses **Facebook Prophet** for time series forecasting, integrated with **MLflow (via DagsHub)** for experiment tracking and **DVC** for data version control.

The goal is to automate a reproducible ML pipeline that:
- Cleans and preprocesses raw sales data
- Trains a time series forecasting model
- Logs experiments, metrics, and artifacts automatically
- Saves the trained Prophet model for reuse or deployment

---

## 🚀 Key Features
✅ End-to-end automated pipeline  
✅ Cleaned and processed over **64K+ sales records**  
✅ Integrated with **MLflow (DagsHub)** for experiment tracking  
✅ Data versioning using **DVC**  
✅ Visual forecast plots and metrics logging  
✅ Modular structure with reusable components  

---

## 🧰 Tools & Technologies

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.10 |
| **Libraries** | Prophet, Pandas, Joblib, MLflow, YAML, Dotenv |
| **Experiment Tracking** | MLflow (via DagsHub) |
| **Data Versioning** | DVC |
| **Visualization** | Matplotlib |
| **IDE** | VS Code |
| **Version Control** | Git & GitHub |

---

## 🧾 Project Structure

```

Sales-Demand-Forecasting/
│
├── data/
│   ├── raw/               <- Original dataset
│   ├── processed/         <- Cleaned data
│   └── interim/           <- Train/Test splits
│
├── models/                <- Trained Prophet model + forecast plot
│
├── src/
│   ├── components/
│   │   ├── data_preprocessing.py
│   │   ├── make_dataset.py
│   │   ├── model_trainer.py
│   │   └── **init**.py
│   │
│   ├── utils/
│   │   └── logger.py
│   │
│   └── main.py            <- Runs the full pipeline
│
├── params.yaml            <- Model configuration
├── requirements.txt
├── .env                   <- Environment variables (MLflow, credentials)
├── dvc.yaml               <- Data versioning configuration
├── README.md
└── scripts/
└── evaluate_predictions.py

````

---

## ⚙️ How to Run

### 🔹 Step 1: Create Environment
```bash
conda create -n sales_forecast_prophet python=3.10 -y
conda activate sales_forecast_prophet
````

### 🔹 Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Step 3: Run Full Pipeline

```bash
python -m src.main
```

### 🔹 Step 4: Evaluate Predictions (optional)

```bash
python scripts/evaluate_predictions.py
```

---

## 📈 Sample Results

| Metric       | Score      |
| ------------ | ---------- |
| **MAE**      | 137,644.37 |
| **RMSE**     | 173,429.71 |
| **R²**       | -0.0115    |
| **MAPE**     | 18.84%     |
| **Accuracy** | 81.16%     |

🧩 **Model file:** `models/prophet_model.pkl`
📊 **Forecast plot:** `models/forecast_plot.png`

---

## 🌐 MLflow Experiment Tracking

All experiments are tracked remotely using **DagsHub MLflow**.

👉 [View MLflow Dashboard](https://dagshub.com/Bishal-Nengminja/Sales-Demand-Forecasting.mlflow)

Includes:

* Run metrics (MAE, RMSE, R², MAPE)
* Model artifacts
* Forecast plots
* Parameters and experiment logs

---

## 🧠 Insights

* Prophet effectively models **seasonal and trend** behavior in sales data.
* MLflow integration enables **experiment reproducibility and comparison**.
* DVC ensures **version-controlled datasets** and model reproducibility.
* Modular structure allows easy **pipeline extension** and automation.

---

## 🧭 Future Improvements

* Add external regressors (e.g., region, channel) to improve accuracy
* Integrate a **Streamlit dashboard** for live forecasts
* Automate model retraining with **GitHub Actions (CI/CD)**
* Store artifacts in **AWS S3 or GCS** for deployment pipelines

---

## 🏷️ Author

**👨‍💻 Bishal Nengminja**
📧 *[bishalnengminja61@gmail.com](bishalnengminja61@gmail.com)*
🔗 [GitHub Profile](https://github.com/Bishal-Nengminja)
🔗 [LinkedIn Profile](https://www.linkedin.com/in/bishal-nengminja/)

---

⭐ **If you find this project helpful, please give it a star on GitHub!**
