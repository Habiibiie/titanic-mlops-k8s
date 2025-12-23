# 🚢 End-to-End Titanic MLOps Project

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

A production-ready Machine Learning pipeline for predicting Titanic survival, deployed on **Kubernetes** using **Docker** and **FastAPI**.

## 🚀 Features
* **Modular Pipeline:** Refactored Jupyter Notebooks into a scalable `src` architecture (Data Ingestion, Transformation, Training).
* **Rest API:** High-performance model serving with **FastAPI** & **Pydantic** validation.
* **Containerization:** Fully Dockerized application ensuring consistency across environments.
* **Orchestration:** Deployed on **Minikube (Kubernetes)** with custom Deployment & Service manifests.
* **Frontend:** Interactive dashboard built with **Streamlit**.
* **CI/QA:** Automated testing with `pytest`.

---

## 🛠️ Tech Stack
* **ML:** Scikit-learn, Pandas, Joblib
* **Backend:** FastAPI, Uvicorn
* **Infrastructure:** Docker, Kubernetes (Minikube)
* **Testing:** Pytest

---

## 📦 How to Run

### 1. Run with Docker
```bash
docker build -t titanic-api:v1 .
docker run -p 8000:8000 titanic-api:v1
```

### 2. Deploy to Kubernetes
```bash
minikube start
minikube image load titanic-api:v1
kubectl apply -f k8s/deployment.yaml
# Access the API
minikube service titanic-service --url
```

### 3. Run Dashboard
```bash
streamlit run src/ui/dashboard.py
```

