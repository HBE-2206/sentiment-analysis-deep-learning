# Sentiment Analysis using Deep Learning

An English sentiment analysis project developed for **DataPulse Solutions** using the IMDb Movie Reviews dataset and a Bidirectional LSTM (BiLSTM) neural network.

## Project Objective

Build a Deep Learning / NLP model that classifies English text into two sentiment classes:

- **Positive**
- **Negative**

The project provides a foundation that can be integrated into future DataPulse Solutions data-analysis tools.

## Dataset

The project uses the standard **IMDb Movie Reviews** dataset through TensorFlow/Keras.

- 25,000 training reviews
- 25,000 test reviews
- English text
- Binary sentiment labels
- Vocabulary limited to the 10,000 most frequent words

The dataset is downloaded automatically by Keras when the notebook is first executed.

## Model Architecture

```text
Input Text
    ↓
Tokenization / Integer Encoding
    ↓
Padding (250 tokens)
    ↓
Embedding
    ↓
Bidirectional LSTM
    ↓
Dropout
    ↓
Dense (ReLU)
    ↓
Dropout
    ↓
Sigmoid
    ↓
Positive / Negative
```

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

## Model Results

Results obtained from the executed model:

| Metric | Result |
|---|---:|
| Training Accuracy (Epoch 4) | **92.44%** |
| Best Validation Accuracy | **85.70%** |
| Test Accuracy | **84.31%** |
| Test Loss | **0.3685** |

### Real-Time Testing

**Positive example**

> This movie was amazing, emotional and absolutely worth watching!

**Prediction:** Positive  
**Confidence:** 76.54%

**Negative example**

> The service was terrible and I completely regret buying this product.

**Prediction:** Negative  
**Confidence:** 90.28%

## Training Summary

| Epoch | Training Accuracy | Training Loss | Validation Accuracy | Validation Loss |
|---|---:|---:|---:|---:|
| 1 | 74.79% | 0.5056 | 51.82% | 1.0325 |
| 2 | 81.53% | 0.4146 | 85.50% | 0.3433 |
| 3 | 89.50% | 0.2840 | 85.20% | 0.3643 |
| 4 | 92.44% | 0.2129 | 85.70% | 0.3958 |

The training accuracy increased faster than validation accuracy by later epochs, indicating some early signs of overfitting. EarlyStopping is used in the notebook to restore the best validation-loss weights.

## Project Structure

```text
sentiment-analysis-deep-learning/
├── Sentiment_Analysis_IMDb_BiLSTM.ipynb
├── README.md
└── requirements.txt
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/HBE-2206/sentiment-analysis-deep-learning.git
cd sentiment-analysis-deep-learning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open the notebook

Open:

`Sentiment_Analysis_IMDb_BiLSTM.ipynb`

### 5. Run all cells

Run the notebook from top to bottom. It covers dataset loading, exploration, preprocessing, model construction, training, evaluation, performance curves, confusion matrix, and real-time prediction.

## Real-Time Testing

The notebook includes a prediction function that accepts a new English sentence and returns:

- Predicted sentiment
- Confidence
- Positive probability

Example:

```text
Input: I really enjoyed this movie!
Output: Positive
Confidence: XX.XX%
```

## Future Improvements

- Pretrained word embeddings such as GloVe
- Transformer-based models such as BERT
- Improved text preprocessing
- REST API deployment
- Web interface
- Multi-class sentiment analysis

## Author

**Hazem Elshafey**
