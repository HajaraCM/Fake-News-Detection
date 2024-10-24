# **Fake News Detection Project**

## **Overview**

The Fake News Detection Project aims to build a machine learning-based web application to detect fake news articles. Using advanced Natural Language Processing (NLP) techniques and various machine learning algorithms, we can classify news articles as either **real** or **fake**. The project pipeline covers end-to-end development, including data preprocessing, feature extraction, model training, evaluation, and deployment through a Flask web API.

## **Table of Contents**

- [Problem Statement](#problem-statement)
- [Technologies and Libraries](#technologies-and-libraries)
- [Data Preprocessing](#data-preprocessing)
  - [Text Cleaning](#text-cleaning)
  - [Tokenization](#tokenization)
  - [Lemmatization](#lemmatization)
  - [Word2Vec](#word2vec)
- [Machine Learning Models](#machine-learning-models)
- [Experiment Tracking (MLflow)](#experiment-tracking-mlflow)
- [Flask API Development](#flask-api-development)
- [Docker Containerization](#docker-containerization)
- [How to Run the Project](#how-to-run-the-project)
- [Conclusion](#conclusion)

---

## **Problem Statement**

With the rapid spread of information across digital platforms, distinguishing between real and fake news has become a major concern. Fake news can have harmful effects on society by spreading misinformation. This project aims to solve this issue by developing a reliable and efficient fake news detection model using machine learning and NLP techniques.

---

## **Technologies and Libraries**

The project leverages several key technologies and libraries to ensure efficient data processing, model development, and deployment:

- **Python**: The core programming language.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **Scikit-learn**: Machine learning models and utilities.
- **MLflow**: To track and manage machine learning experiments.
- **Flask**: For building a web-based API.
- **HTML/CSS**: For the front-end of the web application.
- **Docker**: For containerization and seamless deployment.
- **Word2Vec**: For feature extraction and vectorization of text data.
- **Matplotlib & Seaborn**: For visualizations.
- **NLP Libraries**: Including `nltk` and `spacy` for text preprocessing.

---

## **Data Preprocessing**

Proper preprocessing of textual data is crucial for achieving accurate predictions in any NLP-based project. For this fake news detection project, the following preprocessing techniques were employed:

### **Text Cleaning**

- Removed unnecessary characters such as punctuation, numbers, and special symbols.
- Converted all text to lowercase to maintain uniformity.
- Eliminated stop words (e.g., "the", "is", "and") that don't contribute meaningfully to the text classification task.

### **Tokenization**

- Used tokenization to break down the news articles into individual words (tokens).
- Tokenization helps transform sentences into their constituent words, which is a required step for further NLP processing.

### **Lemmatization**

- Employed lemmatization to reduce words to their base form (lemma). For instance, "running" becomes "run", and "better" becomes "good".
- Lemmatization helps capture the intended meaning of words without unnecessary complexity, improving feature extraction.

### **Word2Vec**

- Implemented the **Word2Vec** model to convert the text into numerical vectors. 
- Word2Vec captures the semantic meaning of words by representing them as vectors in continuous space, where similar words have similar vectors. This technique was pivotal for representing our cleaned and preprocessed text in a way that machine learning algorithms could understand.

---

## **Machine Learning Models**

Several machine learning algorithms were used to build and compare different classifiers for fake news detection. The following models were implemented:

- **Logistic Regression**: A simple yet effective baseline algorithm.
  ```python
  LogisticRegression()
