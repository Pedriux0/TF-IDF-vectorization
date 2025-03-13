# Document Recommender System

This project is a **document recommendation system** that utilizes **TF-IDF vectorization** and **cosine similarity** to suggest relevant articles based on user input. The system is designed to provide a mix of highly similar, medium-similar, and diverse recommendations.

## Features
- Loads preprocessed text data from a CSV file.
- Converts documents into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)**.
- Computes document similarity using **cosine similarity**.
- Provides recommendations categorized into:
  - Highly similar articles
  - Medium similarity articles
  - Diverse recommendations
- Interactive CLI for selecting documents and viewing recommendations.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.7). Install dependencies using:
```bash
pip install -r requirements.txt
```

### Required Libraries
The project uses the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`

You can install them manually if needed:
```bash
pip install pandas numpy scikit-learn
```

## Usage
1. **Prepare Data:** Ensure `docs_stage_3_preprocessed.csv` is available in the project directory.
2. **Run the script:** Execute the following command:
   ```bash
   python recommender.py
   ```
3. **Follow CLI instructions:** Enter an article index to get recommendations, or type `exit` to quit.

## File Structure
```
📁 Document-Recommender-System
│-- recommender.py          # Main script
│-- docs_stage_3_preprocessed.csv # Preprocessed document data
│-- requirements.txt        # Dependencies
│-- README.md               # Project documentation
```

## Example Output
```
Loaded 1000 articles.
Choose the index of the article (0-999) or type exit:
> 5

=== You article is ===
DocId: A123 | DocType: Research Paper
File Size: 512 KB | File Path: /data/research/A123.pdf

=== Highly Similar Articles ===
DocId: B456 | DocType: Research Paper
Snippet: This paper discusses...
...
```

## Contributing
1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature-new-improvement
   ```
3. **Commit changes**
   ```bash
   git commit -m "Added new feature"
   ```
4. **Push to GitHub**
   ```bash
   git push origin feature-new-improvement
   ```
5. **Submit a pull request**

## License
This project is licensed under the MIT License.
