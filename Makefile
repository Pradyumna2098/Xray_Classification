# Makefile for X-Ray Classification Project

.PHONY: help install test lint docker-build docker-run docker-compose-up docker-compose-down k8s-deploy clean

help:
	@echo "Available commands:"
	@echo "  make install          - Install Python dependencies"
	@echo "  make test            - Run tests with coverage"
	@echo "  make lint            - Run code quality checks"
	@echo "  make docker-build    - Build Docker image"
	@echo "  make docker-run      - Run Docker container"
	@echo "  make docker-compose-up   - Start all services with Docker Compose"
	@echo "  make docker-compose-down - Stop all services"
	@echo "  make k8s-deploy      - Deploy to Kubernetes"
	@echo "  make clean           - Clean temporary files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest --cov=src --cov=api --cov-report=html --cov-report=term-missing -v

lint:
	flake8 src/ api/ tests/ --count --statistics || true
	pylint src/ api/ --exit-zero || true

docker-build:
	docker build -t xray-classification-api:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models xray-classification-api:latest

docker-compose-up:
	docker-compose up -d
	@echo "Services started:"
	@echo "  API: http://localhost:8000"
	@echo "  API Docs: http://localhost:8000/docs"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana: http://localhost:3000"

docker-compose-down:
	docker-compose down

k8s-deploy:
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/deployment.yaml
	@echo "Waiting for deployment..."
	kubectl rollout status deployment/xray-api

helm-install:
	helm install xray-api ./helm/xray-classification

helm-upgrade:
	helm upgrade xray-api ./helm/xray-classification

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "Cleaned temporary files"

dev:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

format:
	black src/ api/ tests/ || true
	isort src/ api/ tests/ || true
