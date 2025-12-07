# Frontend Dockerfile
FROM python:3.12-slim

# Définir le répertoire de travail
WORKDIR /app

# Définir les variables d'environnement
ENV PIP_DEFAULT_TIMEOUT=300
ENV PYTHONUNBUFFERED=1

# Copier les fichiers de dépendances
COPY pyproject.toml ./

# Installer les dépendances avec pip
RUN pip install --no-cache-dir --timeout 300 \
    python-dotenv==1.0.1 \
    requests==2.32.5 \
    streamlit==1.49.1

# Copier le code de l'application
COPY app.py ./

# Exposer le port Streamlit
EXPOSE 8501

# Commande pour lancer Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
