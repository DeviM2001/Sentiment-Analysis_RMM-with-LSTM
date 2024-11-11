# Sentiment Analysis Using Recurrent Neural Networks (RNNs)

## Authors
- Dr. David Raj Micheal, Department of Mathematics, School of Advanced Sciences, Vellore Institute of Technology, Chennai
- Devi M, Department of Mathematics, School of Advanced Sciences, Vellore Institute of Technology, Chennai

## Overview

This project presents an RNN-based model for binary sentiment analysis on movie reviews. By utilizing Long Short-Term Memory (LSTM) networks, the model captures sequential dependencies and semantic nuances in text data. The paper explores the preprocessing techniques, model architecture, and experimental outcomes, with promising results indicating high accuracy and robustness in sentiment classification.

## Motivation

Sentiment analysis, also known as opinion mining, is crucial in today's digital landscape, as it provides insights into customer feedback, social trends, and public opinion. Traditional methods like Naive Bayes or rule-based lexicons have limitations in handling complex language structures. RNNs, specifically LSTMs, overcome these limitations by learning long-term dependencies, making them ideal for sentiment classification tasks.

## Methodology

### 1. Data Preprocessing
- Text normalization (lowercasing, punctuation removal).
- Tokenization and word embedding with dense vector representations.
- Use of pre-trained embeddings (e.g., Word2Vec, GloVe) to enrich the vocabulary and capture semantic relationships.

### 2. Model Architecture
- The model comprises an embedding layer, LSTM layers with 256 hidden units, and a dense output layer.
- Dropout regularization and batch normalization are applied to reduce overfitting.
- The binary cross-entropy loss function is used, with the Adam optimizer (learning rate: 0.001).

### 3. Training and Hyperparameters
- The model was trained on 25,000 movie reviews over 15 epochs, with a batch size of 64.
- Early stopping was employed to prevent overfitting.
- The training setup achieved efficient convergence and high accuracy.

## Results and Discussion

- **Performance Metrics**: The model achieved an accuracy of 89.7%, with high precision, recall, and F1-scores.
- **Analysis**: The LSTM network, combined with word embeddings, effectively handled sentiment nuances and complex structures, like negations and sarcasm.
- **Limitations**: Reduced accuracy was observed in sentences requiring deep contextual understanding. Future directions include incorporating attention mechanisms and exploring transformer-based models (e.g., BERT).

## Conclusion

The LSTM-based RNN model demonstrated strong performance in sentiment classification, validating its suitability for real-world applications. Future enhancements could include hybrid architectures with attention mechanisms and transformer models to further improve accuracy and contextual understanding.

## Potential Applications
- Automated customer feedback analysis
- Social media sentiment monitoring

## References
1. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural Computation.
2. Mikolov, T., & Chen, K. (2013). *Efficient estimation of word representations in vector space*. arXiv preprint.
3. Pennington, J., Socher, R., & Manning, C. D. (2014). *GloVe: Global vectors for word representation*. EMNLP.

For further details, see the full report.

---

This repository is maintained as part of ongoing research in the field of text-based sentiment analysis using RNNs.
