# Bank Customer Churn Prediction

This project predicts customer churn for a bank using machine learning models like Random Forest, Logistic Regression, and Decision Tree. It tracks experiments, logs models, and visualizes performance using **MLflow**.

---

## Project Structure

```
.
├── dataset/
│   └── Churn_Modelling.csv
├── output/
│   └── (model artifacts, plots, etc.)
├── src/
│   ├── train.py
│   └── (other source files)
├── api/
│   └── main.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── README.md
└── requirements.txt
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/MLOps-Course-Labs.git
cd MLOps-Course-Labs
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

### 3. Prepare the data

Place the `Churn_Modelling.csv` file in the `dataset/` directory.

### 4. Run MLflow Tracking Server (optional)

If you want to use the MLflow UI:

```bash
mlflow ui
```
Then open [http://localhost:5000](http://localhost:5000) in your browser.

### 5. Train models and log experiments

```bash
python src/train.py
```

### 6. Run the API (optional)

To serve predictions via FastAPI:

```bash
uvicorn api.main:app --reload
```
Then visit [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API documentation.

### 7. Run API Tests

Make sure you are in the project root directory:

```bash
pytest
```

### 8. Run with Docker

You can build and run the application using Docker:

```bash
docker build -t churn-prediction-app .
docker run -p 8000:8000 churn-prediction-app
```

This will start the FastAPI server inside a Docker container.  
Visit [http://localhost:8000/docs](http://localhost:8000/docs) to interact with the API.

## Features

- **Data Preprocessing:** Cleans and transforms raw data for modeling.
- **Model Training:** Trains Logistic Regression, Random Forest, and Decision Tree models.
- **Experiment Tracking:** Uses MLflow to log parameters, metrics, models, and artifacts.
- **Visualization:** Saves and logs confusion matrices and other evaluation plots.
- **Reproducibility:** All runs and artifacts are tracked for easy comparison.
- **API:** FastAPI app for serving predictions and health checks.
- **Testing:** Automated tests for API endpoints.
- **Docker Support:** Easily build and deploy the app in a containerized environment.

## MLflow Artifacts

- **Models:** Serialized models for each algorithm.
- **Preprocessing Pipeline:** Saved transformer for consistent inference.
- **Metrics:** Accuracy, precision, recall, F1-score.
- **Plots:** Confusion matrices for each model.

## Usage Notes

- You can change model parameters in `src/train.py`.
- All experiment runs are tracked in MLflow and can be compared in the UI.
- Artifacts and outputs are saved in the `output/` directory.
- The API expects input data in the same format as the training features.

