
.PHONY: help install init-db sample-data embed run-backend run-frontend run-all test clean docker-up docker-down

help:
	@echo "BluePy - ARGO Data AI Interface"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make init-db       - Initialize database"
	@echo "  make sample-data   - Generate sample data"
	@echo "  make embed         - Generate embeddings"
	@echo "  make run-backend   - Start backend API"
	@echo "  make run-frontend  - Start frontend"
	@echo "  make run-all       - Start both backend and frontend"
	@echo "  make test          - Run tests"
	@echo "  make clean         - Clean temporary files"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make docker-down   - Stop Docker services"

install:
	pip install -r requirements.txt

init-db:
	python scripts/init_db.py

sample-data:
	python scripts/sample_data_generator.py

embed:
	python scripts/embed_profiles.py

run-backend:
	uvicorn backend.main:app

run-frontend:
	streamlit run frontend/app.py

run-all:
	@echo "Starting backend and frontend..."
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:8501"
	@start cmd /k "uvicorn backend.main:app
	@start cmd /k "streamlit run frontend/app.py

test:
	pytest tests/ -v

test-integration:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

format:
	black backend/ ingestion/ frontend/ tests/
	isort backend/ ingestion/ frontend/ tests/

lint:
	flake8 backend/ ingestion/ frontend/
	mypy backend/ ingestion/
