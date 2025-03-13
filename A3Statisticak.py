import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load Data
def load_data():
    # Path to the preprocessed CSV
    preprocessed_text_path = "docs_stage_3_preprocessed.csv"

    # Data read
    try:
        preprocessed_df = pd.read_csv(preprocessed_text_path)
    except FileNotFoundError as e:
        print("The files are not correctly loaded try to verify again {e}")
        exit()

    # Rename columns to match
    preprocessed_df.rename(columns={'DocText': 'FinalText'}, inplace=True)


    # Return the preprocessed
    return preprocessed_df

# Step 2: Convert Documents to Vectors

#Source of info: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#how use the text vectorization


def vectorize_documents(texts):
    # stop_word to improve performance and avoiding too frequent words and less than 2
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9, min_df=2, ngram_range=(1,2))

    # Apply the vectorizer to the input texts and convert them to vectors
    vectors = vectorizer.fit_transform(texts)

    # Return vectors
    return vectors

# Step 3: Compute Similarity
def compute_similarity(vectors):
    # Source of info:cosine_similarity: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html


    # Calculate and return the cosine similarity
    return cosine_similarity(vectors)

# Step 4: Generate Recommendations
def get_recommendations(index, similarity_matrix, df, preprocessed_df, top_n=4, medium_n=3, diverse_n=3):
    # Get the similarity scores for the selected document in the index
    scores = list(enumerate(similarity_matrix[index]))

    # Sort the scores in descending order similar ones first
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Get the type and ID
    selected_doc_type = df.iloc[index]['DocType']
    selected_doc_title = df.iloc[index]['DocId']

    # Get top highly similar articles
    top_similar = [idx for idx, score in scores[1:top_n+1]]  # Skip the element itself

    # Get medium-similarity articles
    mid_range = scores[len(scores)//4:len(scores)//2]
    medium_similar = np.random.choice([idx for idx, _ in mid_range], size=min(medium_n, len(mid_range)), replace=False)

    # Get diverse recommendations
    diverse_candidates = [(idx, score) for idx, score in scores if df.iloc[idx]['DocType'] != selected_doc_type]
    diverse_candidates = sorted(diverse_candidates, key=lambda x: x[1], reverse=True)[:diverse_n * 2]
    # Take top 2x diverse options
    diverse_selected = np.random.choice([idx for idx, _ in diverse_candidates], size=min(diverse_n, len(diverse_candidates)), replace=False)

    # Combine all the selected indices no duplicates
    recommended_indices = list(dict.fromkeys(top_similar + list(medium_similar) + list(diverse_selected)))

    # Get the recommended documents from the DocId and DocType
    recommended_articles = df.iloc[recommended_indices][['DocId', 'DocType']]

    # The first 120 characters
    recommended_articles['Snippet'] = recommended_articles['DocId'].apply(
        lambda x: preprocessed_df[preprocessed_df['DocId'] == x]['FinalText'].values[0][:120] + "..."
    )

    # Return the recommended articles
    return recommended_articles, top_similar, medium_similar, diverse_selected

# Step 5: Print Recommendations in a Clean Format
def print_recommendations(recommendations, top_similar, medium_similar, diverse_selected):
    print("\n===-------- Highly Similar Articles ------===")
    for idx in top_similar:
        article = recommendations[recommendations.index == idx]
        print(f"DocId: {article['DocId'].values[0]} | DocType: {article['DocType'].values[0]}")
        print(f"Snippet: {article['Snippet'].values[0]}")
        # Print lines to separate
        print("-" * 90)

    print("\n===-------- Medium Similarity Articles -------- ===")
    for idx in medium_similar:
        article = recommendations[recommendations.index == idx]
        print(f"DocId: {article['DocId'].values[0]} | DocType: {article['DocType'].values[0]}")
        print(f"Snippet: {article['Snippet'].values[0]}")
        # Print lines to separate
        print("-" * 90)

    print("\n=== --------  Diverse Articles -------- ===")
    for idx in diverse_selected:
        article = recommendations[recommendations.index == idx]
        print(f"DocId: {article['DocId'].values[0]} | DocType: {article['DocType'].values[0]}")
        print(f"Snippet: {article['Snippet'].values[0]}")
        # Print lines to separate
        print("-" * 90)

def main():
    df = load_data()
    print("Loaded", len(df), "articles.")

    # Vectorization
    vectors = vectorize_documents(df['FinalText'])
    similarity_matrix = compute_similarity(vectors)

    while True:
        # Let the user pick
        print("\nChoose the index of the article (0-", len(df)-1, ") or you can type exit to go out")
        user_input = input()
        if user_input.lower() == 'exit':
            break

        # Exceptions of the input
        try:
            user_index = int(user_input)
            if user_index < 0 or user_index >= len(df):
                print("That not a valid number", len(df)-1)
                continue
        except ValueError:
            print("Invalid text do it again")
            continue

        # Information about the article
        selected_article = df.iloc[user_index]
        print("\n=== You article is ===")
        print(f"DocId: {selected_article['DocId']} | DocType: {selected_article['DocType']}")
        print(f"File Size: {selected_article['FileSize']} | File Path: {selected_article['FilePath']}")
        print("\n=== Article Preview ===")
        print(selected_article['FinalText'][:350] + "...")  # The first 350 characters of the text
        print("=" * 80)

        # Get recommendations
        recommendations, top_similar, medium_similar, diverse_selected = get_recommendations(user_index, similarity_matrix, df, df)

        # Print recommendations in a clean format
        print_recommendations(recommendations, top_similar, medium_similar, diverse_selected)

        #Loop
        print("\nTell me you want do it again say 'yes' or 'no'")
        if input().lower() != 'yes':
            break

if __name__ == "__main__":
    main()