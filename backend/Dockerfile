# Backend Dockerfile
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Définir les variables d'environnement pour optimiser l'installation
ENV PIP_DEFAULT_TIMEOUT=300
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Copier les fichiers de dépendances
COPY pyproject.toml ./

# Installer les dépendances légères d'abord (mise en cache efficace)
RUN pip install --no-cache-dir --timeout 300 \
    python-dotenv==1.0.1 \
    fastapi==0.116.1 \
    joblib==1.5.2 \
    pydantic==2.11.7 \
    pymysql==1.1.2 \
    sqlalchemy==2.0.43 \
    uvicorn==0.35.0 \
    cryptography==45.0.7

# Installer PyTorch CPU uniquement (très lourd ~200MB au lieu de 900MB)
# Force CPU-only version sans dépendances CUDA
RUN pip install --no-cache-dir --timeout 300 \
    torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu

# Installer les packages ML lourds
RUN pip install --no-cache-dir --timeout 300 \
    scikit-learn==1.7.2 \
    transformers==4.56.1 \
    peft==0.17.1

# Copier le code de l'application (fait en dernier pour optimiser le cache)
COPY main.py database.py ./

# Créer le répertoire pour les modèles ML (sera monté en volume)
RUN mkdir -p /app/ml/models /app/ml/artifacts

# Exposer le port de l'API
EXPOSE 8000

# Commande pour lancer l'application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
