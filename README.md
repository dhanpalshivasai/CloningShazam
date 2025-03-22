# Enhancing Search Engine Relevance for Video Subtitles (Cloning Shazam)

## Overview
This project aims to improve search relevance for video subtitles by leveraging natural language processing (NLP) and machine learning (ML) techniques. It enhances accessibility by developing an advanced search engine that retrieves subtitles based on user queries. The project supports both keyword-based and semantic search approaches.

## Features
- **Keyword-Based Search:** Uses TF-IDF and Bag-of-Words (BOW) for exact keyword matching.
- **Semantic Search:** Utilizes BERT-based SentenceTransformers for deeper contextual understanding.
- **Document Chunking:** Implements chunking with overlapping windows to optimize embeddings.
- **Cosine Similarity Calculation:** Measures relevance between subtitle embeddings and user queries.
- **ChromaDB Storage:** Stores and retrieves embeddings efficiently.
- **Audio Query Processing:** Converts user audio queries into text for subtitle retrieval.
- **Jupyter Notebook Implementation:** The complete project is implemented in a Jupyter Notebook (`CloningShazam.ipynb`).

## Data
The project uses subtitle data stored in a database file. The dataset can be accessed and processed following these steps:
1. Read and decode the database file.
2. Extract subtitle documents.
3. Apply necessary cleaning (e.g., remove timestamps).
4. Store cleaned data for further processing.

## Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- SentenceTransformers
- ChromaDB
- PyTorch (if using GPU acceleration)
- Jupyter Notebook
- whisper
- 
## Technologies Used

### Programming Language
- **Python 3.8+**

### Machine Learning & NLP
- **SentenceTransformers** – BERT-based embeddings for semantic search
- **Scikit-learn** – TF-IDF, Cosine Similarity for keyword-based search
- **NumPy & Pandas** – Data processing and manipulation
- **NLTK** – Tokenization, stopword removal

### Vector Storage & Retrieval
- **ChromaDB** – Efficient embedding storage and retrieval

### Audio Processing
- **Whisper** – Converts audio queries to text using deep learning
- **PyTorch** – GPU acceleration for ML models

### Development & Execution
- **Jupyter Notebook** – Interactive development and experimentation

## Architecture
1. **Data Ingestion:** Reads and cleans subtitle data.
2. **Vectorization:** Converts text into embeddings using TF-IDF or SentenceTransformers.
3. **Storage:** Saves embeddings in ChromaDB for efficient retrieval.
4. **Query Processing:** Converts user queries (audio) into text using Whisper then into embeddings.
5. **Similarity Matching:** Uses cosine similarity to find the most relevant subtitles.
