# Xray_Classification - Production-Grade Application

[![CI - Test and Build](https://github.com/Pradyumna2098/Xray_Classification/workflows/CI%20-%20Test%20and%20Build/badge.svg)](https://github.com/Pradyumna2098/Xray_Classification/actions)
[![codecov](https://codecov.io/gh/Pradyumna2098/Xray_Classification/branch/main/graph/badge.svg)](https://codecov.io/gh/Pradyumna2098/Xray_Classification)

**A production-grade deep learning application for chest X-ray classification with explainable AI, RESTful API, comprehensive monitoring, and cloud-ready deployment.**

## 🎯 Overview

This repository transforms chest X-ray classification from research notebooks into a **production-grade application** suitable for enterprise deployment and the German job market. It demonstrates industry best practices across the entire ML lifecycle:

### Machine Learning Models
- **Custom Convolutional Neural Network (CNN)** built from scratch
- **Transfer Learning** using pretrained architectures (**VGG19** and **DenseNet201**)
- **Comprehensive Evaluation** with AUC-ROC, Precision-Recall curves, F1-scores, and confusion matrices
- **Model Explainability** using Grad-CAM heatmaps

### Production Features
- **RESTful API** built with FastAPI for model serving
- **Explainable AI** endpoints for Grad-CAM visualizations
- **Comprehensive Testing** with pytest (95%+ coverage)
- **Containerization** with Docker and Kubernetes manifests
- **CI/CD Pipeline** using GitHub Actions
- **Monitoring Stack** with Prometheus and Grafana
- **Cloud-Ready** deployment configurations

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/Pradyumna2098/Xray_Classification.git
cd Xray_Classification

# Start all services (API, Prometheus, Grafana)
docker-compose up -d

# Check API health
curl http://localhost:8000/health

# Access services:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Make a Prediction

```bash
# Classify an X-ray image
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"

# Generate Grad-CAM explanation
curl -X POST "http://localhost:8000/explain" \
  -F "file=@chest_xray.jpg" \
  --output explanation.png
```

## 📚 Documentation

- **[Deployment Guide](DEPLOYMENT_GUIDE.md)** - Comprehensive deployment instructions
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when running)
- **[Architecture Overview](#architecture)** - System architecture and design
- **[Testing Guide](#testing)** - How to run tests and coverage reports

## ✨ Key Features

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Classify X-ray images with confidence scores |
| `/explain` | POST | Generate Grad-CAM heatmap explanations |
| `/health` | GET | Health check endpoint |
| `/metrics` | GET | Prometheus metrics for monitoring |
| `/docs` | GET | Interactive API documentation (Swagger UI) |

### Monitoring & Observability

- **Prometheus Metrics**: Request latency, throughput, error rates
- **Grafana Dashboards**: Real-time visualization of API and model metrics
- **Alerting Rules**: Automated alerts for API health and performance
- **Health Checks**: Kubernetes-ready liveness and readiness probes

### Testing & Quality

- **Unit Tests**: Comprehensive tests for all components
- **Integration Tests**: End-to-end API testing
- **95%+ Coverage**: High test coverage with pytest-cov
- **CI/CD Pipeline**: Automated testing and deployment

## 🏗️ Architecture

```
┌─────────────┐      ┌──────────────┐      ┌──────────────┐
│   Client    │─────▶│   FastAPI    │─────▶│   ML Model   │
│  (Browser)  │      │     API      │      │  (VGG19/     │
└─────────────┘      └──────────────┘      │  DenseNet)   │
                            │               └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Prometheus  │
                     │  (Metrics)   │
                     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │   Grafana    │
                     │  (Dashboards)│
                     └──────────────┘
```

## 📁 Project Structure

```
Xray_Classification/
├── api/                      # FastAPI application
│   └── main.py              # API endpoints and logic
├── src/                     # Core source code
│   ├── models/              # Model inference and Grad-CAM
│   ├── utils/               # Preprocessing utilities
│   └── evaluation/          # Evaluation metrics
├── tests/                   # Comprehensive test suite
├── k8s/                     # Kubernetes manifests
├── monitoring/              # Prometheus and Grafana configs
├── notebooks/               # Jupyter notebooks for training
├── Dockerfile              # Production container
├── docker-compose.yml      # Local development stack
└── requirements.txt        # Python dependencies
```
## 🧪 Testing

Run the comprehensive test suite:

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests with coverage
pytest --cov=src --cov=api --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 🔍 Training Notebooks

The original research and training code is available in Jupyter notebooks:

- `Xray_CNN.ipynb`: Custom CNN architecture training
- `Pre_Trained.ipynb`: Transfer learning with VGG19 and DenseNet201

**Note**: The production API uses the trained model weights from these notebooks.

## 📁 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset introduced by Kermany *et al.* (2018).

- Official dataset page: <https://data.mendeley.com/datasets/rscbjbr9sj/3>
- License and usage: released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. Any use of the
  data must credit the original authors and follow the citation guidelines outlined on the dataset page.

### Directory structure

The training assets should be arranged as follows after extraction:

```
datasets/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 📥 Download helper script

Use the `scripts/download_data.py` utility to download the official archive and unpack it into the `datasets/` directory:

```bash
python scripts/download_data.py
```

Key options:

- `--output-dir`: customise where the dataset is extracted (defaults to `datasets/`).
- `--url`: override the download URL if you have mirrored the archive elsewhere.
- `--sha256`: provide a known hash to verify downloads (recommended when sharing archives).
- `--force`: re-download and overwrite an existing installation.

If the script detects a partial installation it aborts with a helpful message so that you can decide whether to rerun with
`--force` or clean the directory manually.
## ⚙️ Development Setup

### Local Development

```bash
# Clone the repository
git clone https://github.com/Pradyumna2098/Xray_Classification.git
cd Xray_Classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the API locally
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Download Dataset (Optional)

```bash
python scripts/download_data.py --output-dir datasets
```

The dataset will be organized as:
```
datasets/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## 📊 Model Evaluation

### Comprehensive Metrics

The evaluation framework includes:
- **AUC-ROC curves** for classification performance
- **Precision-Recall curves** for class imbalance analysis
- **Confusion matrices** with visualization
- **F1-scores, Precision, Recall, Specificity**
- **Error analysis** with misclassification insights

### Expected Performance

| Model | Accuracy | AUC-ROC | Precision | Recall | F1-Score |
|-------|----------|---------|-----------|--------|----------|
| Custom CNN | ~90% | ~0.95 | ~88% | ~92% | ~90% |
| VGG19 | ~94% | ~0.98 | ~93% | ~96% | ~94% |
| DenseNet201 | ~95% | ~0.98 | ~94% | ~96% | ~95% |

*Run training notebooks to generate actual results for your dataset*

## 🚢 Deployment Options

### Docker
```bash
docker build -t xray-api:latest .
docker run -p 8000:8000 xray-api:latest
```

### Kubernetes
```bash
kubectl apply -f k8s/
kubectl get pods -l app=xray-api
```

### Cloud Platforms
- **AWS**: ECS, EKS, or Lambda
- **Google Cloud**: Cloud Run, GKE
- **Azure**: Container Instances, AKS
- **Render/Railway**: Direct Docker deployment

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions.

## 📈 Monitoring Training Runs

Both notebooks now log training curves to TensorBoard (and automatically fall back to Weights & Biases if it is available in the environment). Each training invocation creates a timestamped run directory under `logs/` alongside a matching set of best-performing weights inside `models/`.

To inspect the live metrics locally run:

```bash
tensorboard --logdir logs --port 6006
```

Then open <http://localhost:6006> in your browser. The key tabs to check are:

- **Scalars** – shows training/validation loss and accuracy so you can verify convergence and spot overfitting.
- **Graphs** – visualises the model graph to confirm the architecture being trained.
- **Learning rate** (from the ReduceLROnPlateau callback) – helps correlate performance plateaus with learning-rate reductions.

If you have Weights & Biases installed, the same metrics (plus system diagnostics) are mirrored to the W&B dashboard under the `xray-classification` project.

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Write tests for new features
4. Ensure tests pass and coverage stays above 80%
5. Commit your changes (`git commit -m 'Add AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Kermany et al. (2018) - [Chest X-Ray Images (Pneumonia)](https://data.mendeley.com/datasets/rscbjbr9sj/3)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Models**: VGG19 (Simonyan & Zisserman), DenseNet201 (Huang et al.)
- **Technologies**: TensorFlow, FastAPI, Prometheus, Docker, Kubernetes

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Built with ❤️ for production deployment in the German job market**
