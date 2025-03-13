Document Recommender System
Welcome to the Document Recommender System repository! This project is a simple yet powerful tool for recommending similar documents based on their content. It uses TF-IDF vectorization and cosine similarity to find documents that are most similar to a user-selected article. Whether you're working with news articles, scientific abstracts, or any other text data, this system can help you discover relevant content quickly and efficiently.

Features
Text Vectorization: Converts documents into numerical vectors using TF-IDF.

Similarity Calculation: Uses cosine similarity to measure how similar two documents are.

Recommendation Engine: Recommends articles based on their similarity to a user-selected document.

Diverse Recommendations: Includes both highly similar and less similar articles to help users explore new topics.

Interactive Interface: Allows users to select an article and view recommendations in a clean, readable format.

How It Works
Data Loading: The system loads a dataset of documents (e.g., news articles, scientific abstracts).

Vectorization: Each document is converted into a numerical vector using TF-IDF.

Similarity Calculation: The system calculates the cosine similarity between all pairs of documents.

Recommendation Generation: When a user selects a document, the system recommends the most similar articles, along with a few less similar ones for diversity.

Output: The recommendations are displayed in a clean, easy-to-read format, including a snippet of each recommended article.

Getting Started
Prerequisites
Before running the code, make sure you have the following installed:

Python 3.x

Required Python libraries: pandas, numpy, scikit-learn

You can install the required libraries using pip:

bash
Copy
pip install pandas numpy scikit-learn
Usage
Clone the Repository:

bash
Copy
git clone https://github.com/your-username/document-recommender-system.git
cd document-recommender-system
Run the Script:

bash
Copy
python recommender.py
Follow the Prompts:

The program will load the dataset and display the number of articles.

You will be prompted to choose an article by entering its index.

The program will display recommendations, including highly similar and diverse articles.

Dataset
This project uses the BBC News Dataset, which contains news articles from the BBC. The dataset includes the following fields:

DocId: Unique identifier for each article.

DocType: Category of the article (e.g., Business, Tech, Politics).

FinalText: The text content of the article.

You can replace the dataset with your own data by modifying the load_data() function in the code.

Code Structure
recommender.py: The main script that loads the data, vectorizes the documents, computes similarity, and generates recommendations.

docs_stage_3_preprocessed.csv: The dataset containing the preprocessed text of the articles
