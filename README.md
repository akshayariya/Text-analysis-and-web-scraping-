# Text Analysis and Web Scraping Project

## Overview
This repository contains a Python project that performs text analysis on articles scraped from various URLs. The project utilizes web scraping techniques to extract text content from web pages and applies Natural Language Processing (NLP) methods to analyze the extracted text. The analysis includes sentiment analysis, complexity metrics, average sentence length, and more.

## Files
- **main.py**: The main Python script that handles web scraping, text preprocessing, and analysis.
- **Input.xlsx**: Excel file containing URLs from which data will be scraped.
- **all_stopwords.txt**: A file containing a consolidated list of stopwords used for text preprocessing.
- **Output_File.xlsx**: Output Excel file where analysis results are stored.

## Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-akshayariya/text-analysis-web-scraping-.git
   cd text-analysis-web-scraping
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas requests beautifulsoup4 nltk textblob
   ```

3. **Run the script:**
   ```bash
   python main.py
   ```
   This script will perform the following steps:
   - Read URLs from `Input.xlsx`.
   - Perform web scraping on each URL to extract article text.
   - Perform text preprocessing (removing stopwords, lemmatization).
   - Analyze the preprocessed text using NLP techniques (sentiment analysis, complexity metrics).
   - Store the analysis results in `Output_File.xlsx`.

4. **View the output:**
   After running the script, check `Output_File.xlsx` for the results of the text analysis.

## Column Descriptors (Output_File.xlsx)
- **URL_ID**: Unique identifier for each scraped URL.
- **URL**: The URL from which the data was scraped.
- **Positive Score**: Number of positive sentences identified in the article.
- **Negative Score**: Number of negative sentences identified in the article.
- **Polarity Score**: Sentiment polarity score of the article.
- **Subjectivity Score**: Subjectivity score of the article.
- **Avg Sentence Length**: Average length of sentences in the article.
- **Percentage of Complex Words**: Percentage of words identified as complex.
- **Fog Index**: Readability index calculated based on average sentence length and percentage of complex words.
- **Average number of words per sentence**: Average number of words in each sentence.
- **Complex word count**: Total count of words identified as complex.
- **Total word count**: Total number of words in the article.
- **Syllables per word**: Average number of syllables per word in the article.
- **Personal pronoun count**: Count of personal pronouns (e.g., I, we) in the article.
- **Average word length**: Average length of words in the article.

## Notes
- Ensure you have Python installed along with the necessary libraries (`pandas`, `requests`, `beautifulsoup4`, `nltk`, `textblob`).
- The script `main.py` performs web scraping and text analysis based on URLs provided in `Input.xlsx`.
- Results of the analysis are stored in `Output_File.xlsx` for further examination and use.
