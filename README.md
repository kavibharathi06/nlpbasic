# NLP Text Preprocessing Project

## Overview

This project demonstrates fundamental Natural Language Processing (NLP) preprocessing techniques using Python.

The workflow shows how raw text is transformed into machine-readable representations through multiple preprocessing stages.

## Features

* Tokenization
* Stemming
* Lemmatization
* Unique Vocabulary Extraction
* Bag of Words (BoW)
* One-Hot Encoding
* Conceptual Word Embedding Demonstration

## Technologies Used

* Python 3.11
* NLTK
* Scikit-learn

## Project Workflow

1. Input raw text data
2. Perform tokenization
3. Apply stemming and lemmatization
4. Generate unique vocabulary
5. Convert text into Bag of Words representation
6. Demonstrate one-hot encoding
7. Simulate word embeddings

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd NLP
```

Install dependencies:

```bash
pip install nltk scikit-learn
```

Download required NLP resources:

```python
import nltk

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## Run the Project

```bash
python3.11 nlpproject.py
```

## Sample Concepts Demonstrated

* Relationship between similar words
* Root word extraction
* Text vectorization
* Numerical representation of language

## Project Structure

```text
NLP/
│
├── nlpproject.py
├── README.md
└── requirements.txt
```

## Output

The program displays:

* Raw text
* Tokenized text
* Stemmed output
* Lemmatized output
* Vocabulary generation
* Bag of Words matrix
* One-hot vectors
* Word embedding examples


