# 🔍 Fake News Detection using BERT Transformers

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning project that uses BERT (Bidirectional Encoder Representations from Transformers) to classify news articles as real or fake with 95%+ accuracy.

## 🎯 Project Overview

This project implements a binary text classification system using a fine-tuned BERT model to detect fake news articles. The model analyzes the content and linguistic patterns to determine the credibility of news.

## ✨ Features

- Fine-tuned BERT-base-uncased model
- 95%+ accuracy on test dataset
- REST API for real-time predictions
- Interactive web interface
- Comprehensive data preprocessing pipeline
- Detailed model evaluation metrics

## 🏗️ Architecture
Input Text → BERT Tokenizer → BERT Model → Classification Head → Output (Real/Fake)
**Model Details:**
- Base Model: `bert-base-uncased`
- Classification Head: Dense layer (768 → 2)
- Loss Function: Cross Entropy
- Optimizer: AdamW
- Learning Rate: 2e-5

## 📊 Dataset

- **Source**: Kaggle Fake News Detection Dataset
- **Size**: 20,000+ labeled articles
- **Split**: 80% train, 10% validation, 10% test
- **Classes**: Real (0), Fake (1)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detection-bert.git
cd fake-news-detection-bert
```

2. Create virtual environment:
```bash
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training the Model
```bash
python src/train.py --epochs 3 --batch_size 16 --learning_rate 2e-5
```

### Making Predictions
```python
from src.predict import FakeNewsDetector

detector = FakeNewsDetector('models/saved_models/bert_fake_news.pth')
text = "Your news article text here..."
prediction, confidence = detector.predict(text)
print(f"Prediction: {prediction}, Confidence: {confidence:.2f}%")
```

### Running the Web App
```bash
python app/app.py
```
Visit `http://localhost:5000` in your browser.

## 📈 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 95.8%  |
| Precision | 94.2%  |
| Recall    | 96.1%  |
| F1-Score  | 95.1%  |

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### Training Curves
![Training Curves](images/training_curves.png)

## 📁 Project Structure
fake-news-detection-bert/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── init.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── saved_models/
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
└── tests/
└── test_model.py
## 🔬 Methodology

1. **Data Preprocessing**: Text cleaning, tokenization using BERT tokenizer
2. **Model Architecture**: BERT-base-uncased + custom classification head
3. **Training**: Fine-tuning with frozen/unfrozen layers
4. **Evaluation**: Cross-validation, confusion matrix, ROC-AUC
5. **Deployment**: Flask API for inference

## 🎓 Key Learnings

- Transfer learning with pre-trained transformers
- Handling imbalanced datasets
- Text preprocessing for NLP tasks
- Model fine-tuning strategies
- API development for ML models

## 🚧 Future Improvements

- [ ] Multi-language support
- [ ] Explainability with attention visualization
- [ ] Real-time fact-checking integration
- [ ] Mobile app deployment
- [ ] Ensemble with other models (RoBERTa, ELECTRA)

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Fake News Detection Dataset](https://www.kaggle.com/c/fake-news/data)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## 👤 Author

**Your Name**
- GitHub: [](## 🔬 Methodology

1. **Data Preprocessing**: Text cleaning, tokenization using BERT tokenizer
2. **Model Architecture**: BERT-base-uncased + custom classification head
3. **Training**: Fine-tuning with frozen/unfrozen layers
4. **Evaluation**: Cross-validation, confusion matrix, ROC-AUC
5. **Deployment**: Flask API for inference

## 🎓 Key Learnings

- Transfer learning with pre-trained transformers
- Handling imbalanced datasets
- Text preprocessing for NLP tasks
- Model fine-tuning strategies
- API development for ML models

## 🚧 Future Improvements

- [ ] Multi-language support
- [ ] Explainability with attention visualization
- [ ] Real-time fact-checking integration
- [ ] Mobile app deployment
- [ ] Ensemble with other models (RoBERTa, ELECTRA)

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Fake News Detection Dataset](https://www.kaggle.com/c/fake-news/data)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author
**Your Name**
- GitHub: [@neerajakondala3-alt](https://gihttps://github.com/neerajakondala3-altthub.com/yourusername)


## ⭐ Acknowledgments

- Thanks to Hugging Face for the Transformers library
- Kaggle for providing the dataset
- The open-source community

---

If you found this project helpful, please give it a ⭐!e)


## ⭐ Acknowledgments

- Thanks to Hugging Face for the Transformers library
- Kaggle for providing the dataset
- The open-source community

---

If you found this project helpful, please give it a ⭐!


