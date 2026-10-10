# Sentiment Analysis with Deep Learning

A natural language processing project that classifies English movie reviews as **positive** or **negative** using a **Bidirectional LSTM (BiLSTM)** neural network built with TensorFlow/Keras.

> The notebook trains and evaluates the model from the IMDb dataset. Metrics are generated on execution; any older metric values should be treated as historical until reproduced with the current notebook.

## Project Overview

- **Task:** Binary sentiment classification
- **Dataset:** IMDb movie reviews provided by `tf.keras.datasets.imdb`
- **Model:** Embedding → Bidirectional LSTM → Dense layers → Sigmoid
- **Sequence length:** 250 tokens
- **Vocabulary:** 10,000 most frequent word IDs
- **Evaluation:** Held-out test loss, accuracy, ROC-AUC, classification report, and confusion matrix
- **Notebook:** Jupyter / Google Colab compatible

## Repository Structure

```text
sentiment-analysis-deep-learning/
├── Sentiment_Analysis_IMDb_BiLSTM.ipynb  # Full training and evaluation workflow
├── requirements.txt                       # Python dependencies
└── README.md
```

## Getting Started

### Option 1 — Google Colab

Open the [notebook in Google Colab](https://colab.research.google.com/github/HBE-2206/sentiment-analysis-deep-learning/blob/main/Sentiment_Analysis_IMDb_BiLSTM.ipynb), then run the cells from top to bottom. A GPU runtime can reduce training time, but CPU execution is also supported.

### Option 2 — Run locally

```bash
git clone https://github.com/HBE-2206/sentiment-analysis-deep-learning.git
cd sentiment-analysis-deep-learning

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Open `Sentiment_Analysis_IMDb_BiLSTM.ipynb` and run all cells. Keras downloads the IMDb dataset when needed, so the first execution requires an internet connection.

## Workflow

1. Load the IMDb training and test splits.
2. Pad and truncate integer-encoded reviews to a consistent length.
3. Build a BiLSTM binary classifier.
4. Train with a validation split, early stopping, and learning-rate reduction.
5. Plot learning curves.
6. Evaluate against the held-out test set.
7. Display a confusion matrix and classification report.
8. Run inference on custom English text.
9. Save the trained model as `artifacts/imdb_bilstm.keras`.

## Model Architecture

```text
Integer-encoded review
        ↓
Padding / truncation (250 tokens)
        ↓
Embedding (10,000-token vocabulary)
        ↓
Bidirectional LSTM
        ↓
Dense + Dropout
        ↓
Sigmoid probability
        ↓
Negative / Positive
```

## Example Inference

The notebook includes a `predict_sentiment(text)` helper returning the predicted class, confidence, and positive-class probability. Example outputs depend on the model weights produced during the run.

## Important Notes

- The classification scope is binary sentiment in English.
- IMDb vocabulary token IDs and preprocessing must remain consistent between training and inference.
- Validation data is used during training for early stopping; the test set is held out for final evaluation.
- Neural network results can vary with TensorFlow version, hardware, and runtime settings.
- No trained model weights are committed; the notebook produces them when executed.

## Technologies

Python · TensorFlow · Keras · NumPy · Pandas · scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

## Author

**Hazem Elshafey**
