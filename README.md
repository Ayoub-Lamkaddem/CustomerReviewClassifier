# Customer Review Classifier

**Customer Review Classifier** is a machine learning project designed to analyze and classify customer reviews into three sentiment categories: Positive, Negative, and Neutral.

The project leverages advanced NLP techniques, including text preprocessing (cleaning, lemmatization, stopword removal) and powerful models like BERT with LoRA fine-tuning as well as classical machine learning models such as SGDClassifier.

Additionally, the project uses a **MySQL database** to store user reviews and the predictions from the models, making it easier to track and analyze sentiment data over time.

Users can input a review and quickly get a sentiment prediction, making it useful for businesses to understand customer feedback, monitor satisfaction, and make data-driven decisions.

## Data

The original data comes from the [Yelp Open Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset), a rich resource containing millions of reviews.

## Installation

### 1 - Clone the repository
```bash
git clone https://github.com/Ayoub-Lamkaddem/CustomerReviewClassifier.git
cd CustomerReviewClassifier
```

### 2 - Install **uv**
Before anything, install uv depending on your OS:

- **For Windows (PowerShell):**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- **For macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### 3 - Create a virtual environment (recommended) and Install dependencies:
We need three environments: **frontend**, **backend**, and **ml**.

For each folder, run:
- **frontend folder**
```bash
cd frontend
uv init --python 3.12
uv venv
uv sync
cd ..
```
- **backend folder**
```bash
cd backend
uv init --python 3.12
uv venv
uv sync
cd ..
```
- **ML folder**
```bash
cd ml
uv init --python 3.12
uv venv
uv sync
cd ..
```
### 4 - Configure the .env file
Create a **.env** file in the root of your project with the following structure:

```.env
    ### Backend URL

    BACK_END_URL=http://localhost:8000

    ### Machine Learning Models

    BERT_MODEL_PATH=../ml/models/fine_tuning_bert/BERT_LORA

    SGD_MODEL_PATH=../ml/models/sgd_model/SGD.pkl

    TFIDF_PATH=../ml/artifacts/TfidfVectorizer.pkl

    LABEL_ENCODER_PATH=../ml/artifacts/LabelEncoder.pkl

    ### MySQL configuration

    MYSQL_USER=your_mysql_username

    MYSQL_PASSWORD=your_mysql_password

    MYSQL_HOST=localhost

    MYSQL_PORT=your_mysql_port

    MYSQL_DATABASE=your_database_name

    MYSQL_ROOT_PASSWORD=your_mysql_root_password
```

### 5- Build and start the necessary Docker containers
Make sure your .env file is configured, then run:
```bash
docker compose up -d
```

### 6- Activate the virtual environment and run the project:
- **Frontend folder**
```bash
cd frontend
# Windows
    ./venv/scripts/activate

# Linux or Mac
    source .venv/bin/activate

# Run the frontend
    streamlit run app.py
```

-**backend folder**(open another terminal)
```bash
cd backend
# Windows
    ./venv/scripts/activate

# Linux or Mac
    source .venv/bin/activate

# Run the backend
    uvicorn main:app --reload
```




