# ClauseWise - AI Legal Document Analyzer
## Deployment Guide

This guide provides comprehensive instructions for deploying ClauseWise using different methods.

## Quick Start with Gradio

### Prerequisites
- Python 3.8 or higher
- GPU recommended for IBM Granite model (8GB+ VRAM)
- 16GB+ RAM for optimal performance

### Installation

1. **Clone or download the project files**
   ```bash
   git clone <repository-url>
   cd clausewise-legal-analyzer
   ```

2. **Create virtual environment**
   ```bash
   python -m venv clausewise_env

   # Windows
   clausewise_env\Scripts\activate

   # macOS/Linux
   source clausewise_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   # Download spaCy English model
   python -m spacy download en_core_web_sm
   ```

4. **Run the application**
   ```bash
   python clausewise_implementation.py
   ```

5. **Access the application**
   - Local: http://localhost:7860
   - Public link will be generated automatically for sharing

## Production Deployment Options

### Option 1: Heroku Deployment

1. **Prepare Heroku files**

   Create `Procfile`:
   ```
   web: python clausewise_implementation.py --server.port=$PORT --server.address=0.0.0.0
   ```

   Create `runtime.txt`:
   ```
   python-3.9.16
   ```

2. **Deploy to Heroku**
   ```bash
   # Install Heroku CLI
   heroku create your-app-name
   heroku stack:set heroku-20
   git add .
   git commit -m "Deploy ClauseWise"
   git push heroku main
   ```

### Option 2: Docker Deployment

1. **Build and run**
   ```bash
   docker build -t clausewise .
   docker run -p 7860:7860 clausewise
   ```

   Or use docker-compose:
   ```bash
   docker-compose up --build
   ```

### Option 3: Cloud Platform Deployment

#### Google Cloud Platform
```bash
# Deploy to Cloud Run
gcloud run deploy clausewise \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### AWS EC2
```bash
# Launch EC2 instance
# Install dependencies
# Run application with production server
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:8000
```

For support, check the logs in `./logs/clausewise.log` or create an issue in the repository.
