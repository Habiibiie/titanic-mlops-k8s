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

```mermaid
graph TD
    subgraph Training_Pipeline [🏗️ Training Pipeline]
        style Training_Pipeline fill:#f9f9f9,stroke:#333,stroke-width:2px
        RawData[(Titanic CSV)] -->|Ingestion| Components[src/components]
        Components -->|Preprocessing| Features[Feature Engineering]
        Features -->|Train| ModelTrainer[Model Training]
        ModelTrainer -->|Output| ModelArtifact{{model.joblib}}
    end

    subgraph Containerization [🐳 Dockerization]
        style Containerization fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
        ModelArtifact -.->|Copy| FastAPI[FastAPI Application]
        FastAPI -->|Build| DockerImage[Docker Image]
    end

    subgraph Kubernetes_Cluster [☸️ Kubernetes Environment]
        style Kubernetes_Cluster fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
        DockerImage -->|Deploy| K8sDeploy[Deployment]
        K8sDeploy -->|Manage| Pods(Pod: titanic-api)
        Pods -->|Expose| K8sService[Service]
    end

    subgraph User_Interface [💻 Client Side]
        style User_Interface fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
        Streamlit[Streamlit Dashboard]
        User((User))
    end

    User -->|Interacts| Streamlit
    Streamlit -->|HTTP POST /predict| K8sService
```

---

## 📂 Project Structure
The project follows a modular architecture to separate concerns (Training vs. Inference):

```text
titanic-mlops-k8s/
├── .github/
│   └── workflows/
│       └── deploy.yml      # CI/CD Pipeline for AWS Deployment
├── k8s/                     # Kubernetes Manifests (Deployment & Service)
├── src/
│   ├── api/                 # FastAPI Application (Entry point)
│   ├── components/          # ML Pipeline Components (Ingestion, Transformation)
│   ├── pipelines/           # Training Pipelines
│   └── ui/                  # Streamlit Dashboard Code
├── tests/                   # Pytest Unit Tests
├── .dockerignore            
├── .gitignore
├── Dockerfile               # Multi-stage Docker Build
├── docker-compose.yml       # Container Orchestration (Local & Prod) 
├── requirements.txt         # Project Dependencies
└── params.yaml              # Configuration Controller
```

---

## 📦 How to Run

### 1. Load the project
```bash
git clone https://github.com/enesgulerml/titanic-mlops-k8s.git
cd titanic-mlops-k8s
```

### 2. Local Development (Python)
If you want to run the components separately for development:
```bash
# Start Backend
uvicorn src.api.app:app --reload

# Start Frontend (In another terminal)
streamlit run src/ui/dashboard.py
```

### 3. Local Deployment (Docker Compose) - RECOMMENDED
The easiest way to run the entire stack (API + UI) with a single command:
```bash
# Build and Start
docker-compose up --build
```
Access the Dashboard at: http://localhost:8501

### 4. Kubernetes (Local Test)
To test the orchestration layer locally:
```bash
minikube start
kubectl apply -f k8s/
kubectl get pods
```


