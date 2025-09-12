# Customer Review Classifier
**Customer Review Classifier** is a machine learning project designed to analyze and classify customer reviews into three sentiment categories: Positive, Negative, and Neutral.

The project leverages advanced NLP techniques, including text preprocessing (cleaning, lemmatization, stopword removal) and powerful models like BERT with LoRA fine-tuning as well as classical machine learning models such as SGDClassifier.

Users can input a review and quickly get a sentiment prediction, making it useful for businesses to understand customer feedback, monitor satisfaction, and make data-driven decisions.
## Data

The original data comes from the [Yelp Open Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset), a rich resource containing millions of reviews.

## Installation
### 1-Clone the repository:
```bash
git clone https://github.com/Ayoub-Lamkaddem/CustomerReviewClassifier.git
cd CustomerReviewClassifier
```
Before anything, you need to install **uv**. Run the following command based on your operating system:

- **For Windows (PowerShell):**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- **For macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### 2- Create a virtual environment (recommended):
```bash
uv venv --python 3.12
```

#### Activate the virtual environment:
- Windows

    ```powershell
    ./venv/scripts/activate
    ```

- Linux or Mac

    ```bash
    source .venv/bin/activate
    ```

### 3- Install dependencies:
```bash
uv sync
```

### 4- Launch the Streamlit application
```bash
streamlit run app.py
```
## Deployment
This project has been **deployed** and **hosted** using [Streamlit Community Cloud](https://share.streamlit.io/)

==> You can try the live demo here:
[Customer Review Classifier App](https://customerreviewclassifier-app.streamlit.app/)






