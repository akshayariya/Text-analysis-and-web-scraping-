import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import re
import os

nltk.download('punkt')
nltk.download('wordnet')

#####################################################################

# Specify the file path
input_file = "/content/drive/MyDrive/Blackcoffer/Blackcoffer ass/Input.xlsx"

# Read URLs from input.xlsx
df_urls = pd.read_excel(input_file)

# Display the first few rows of the DataFrame to verify the data
print(df_urls.head())


# Load stop words with robust error handling for encoding issues
def load_stopwords(filename):
    """Loads stop words from a single file, handling potential encoding issues.

    Args:
        filename: The path to the stop word file.

    Returns:
        A set of stop words loaded from the file.
    """
    stopwords = set()
    try:
        # Open with multiple encoding attempts
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                stopwords.add(line.strip().lower())
    except UnicodeDecodeError:
        # Try latin-1 if UTF-8 fails
        try:
            with open(filename, 'r', encoding='latin-1') as file:
                for line in file:
                    stopwords.add(line.strip().lower())
        except UnicodeDecodeError:
            print(f"Warning: Couldn't decode {filename} using UTF-8 or latin-1 encoding.")
    return stopwords

# Define the path to the file containing stop words
stopword_filename = '/content/drive/MyDrive/Blackcoffer/Blackcoffer ass/StopWords-20240529T134840Z-001/StopWords/StopWords_Auditor.txt'
stopword_filename = '/content/drive/MyDrive/Blackcoffer/Blackcoffer ass/StopWords-20240529T134840Z-001/StopWords/StopWords_Currencies.txt'
stopword_filename = '/content/drive/MyDrive/Blackcoffer/Blackcoffer ass/StopWords-20240529T134840Z-001/StopWords/StopWords_DatesandNumbers.txt'
stopword_filename = '/content/drive/MyDrive/Blackcoffer/Blackcoffer ass/StopWords-20240529T134840Z-001/StopWords/StopWords_Generic.txt'
stopword_filename = '/content/drive/MyDrive/Blackcoffer/Blackcoffer ass/StopWords-20240529T134840Z-001/StopWords/StopWords_GenericLong.txt'
stopword_filename = '/content/drive/MyDrive/Blackcoffer/Blackcoffer ass/StopWords-20240529T134840Z-001/StopWords/StopWords_Geographic.txt'
stopword_filename = '/content/drive/MyDrive/Blackcoffer/Blackcoffer ass/StopWords-20240529T134840Z-001/StopWords/StopWords_Names.txt'

# Define the path to the output file
output_filepath = 'all_stopwords.txt'

# Load stop words from the specified file
stopwords = load_stopwords(stopword_filename)

# Append the stop words to the output file
with open(output_filepath, 'a', encoding='utf-8') as output_file:
    for word in stopwords:
        output_file.write(word + '\n')

print("Stop words appended to:", output_filepath)

####################################################################

# Load the input Excel file
df = pd.read_excel(input_file)
def scrape_data(url):
    """Scrapes data from a website and returns the text content.

    Args:
        url: The URL of the website to scrape.

    Returns:
        The text content of the article.
    """
    # Send an HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the elements containing the desired data
        data_elements = soup.find_all('p')  # Example selector

        # Remove the last two <p> tags
        if len(data_elements) > 3:
            data_elements = data_elements[:-3]

        # Extract and return the data
        article_text = ''
        for element in data_elements:
            article_text += element.text.strip() + '\n'

        #print article text
        # print(article_text)

        # Call analysis function
        analysis_result = perform_text_analysis(article_text)
        # Print result analysis
        print(analysis_result)
        print("\n")

        return article_text, analysis_result  # Return the article text and analysis result
    else:
        print(f"Error: Failed to download the page. Status code: {response.status_code}")
        return None, None

# Define an empty list to store the analysis results
analysis_results = []
url_ids = []

# Scrape data for the URLs
urls = df['URL'][:101]  # Adjust the number of URLs as needed

for i, url in enumerate(urls):
    print(f"Scraping URL {i+1}: {url}")
    article_text, analysis_result = scrape_data(url)

    if analysis_result:
        # Generate the URL_ID
        url_id = f"blackassign{i+1:04d}"  # Adjust the format as needed
        url_ids.append(url_id)

        # Append the analysis result to the list
        analysis_results.append({'URL_ID': url_id, 'URL': url, **analysis_result})

# Convert the list of dictionaries into a DataFrame
analysis_df = pd.DataFrame(analysis_results)

# Define the path for the output Excel file
output_file = "/content/drive/MyDrive/Blackcoffer/Blackcoffer ass/Copy of Output Data Structure.xlsx"

# Write the DataFrame to the output Excel file
analysis_df.to_excel(output_file, index=False)

###########################################################################


# Load stop words from the all_stopwords.txt file
def load_stopwords(filename):
    stopwords_set = set()
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                stopwords_set.add(line.strip().lower())
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
    return stopwords_set

# Define the filename of the all_stopwords.txt file
all_stopwords_filename = 'all_stopwords.txt'

# Load the stop words from all_stopwords.txt
stop_words = load_stopwords(all_stopwords_filename)

# Function to clean and preprocess text
def preprocess_text(text):
    words = word_tokenize(text.lower())

    # Remove stopwords using the loaded stop words
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Function to calculate sentiment scores
def calculate_sentiment_scores(text):
    blob = TextBlob(text)
    polarity_score = blob.sentiment.polarity
    subjectivity_score = blob.sentiment.subjectivity
    positive_score = sum(1 for sentence in blob.sentences if sentence.sentiment.polarity > 0)
    negative_score = sum(1 for sentence in blob.sentences if sentence.sentiment.polarity < 0)

    return positive_score, negative_score, polarity_score, subjectivity_score

# Function to calculate average sentence length
def calculate_avg_sentence_length(text):
    sentences = sent_tokenize(text)
    total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
    total_sentences = len(sentences)
    avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
    return avg_sentence_length

# Function to calculate percentage of complex words
def calculate_percentage_complex_words(text):
    words = preprocess_text(text)
    num_complex_words = sum(1 for word in words if len(word) > 2)
    total_words = len(words)
    percentage_complex_words = (num_complex_words / total_words) * 100 if total_words > 0 else 0
    return percentage_complex_words

# Function to calculate Fog index
def calculate_fog_index(avg_sentence_length, percentage_complex_words):
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    return fog_index

# Function to calculate average number of words per sentence
def calculate_avg_words_per_sentence(text):
    sentences = sent_tokenize(text)
    total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
    total_sentences = len(sentences)
    avg_words_per_sentence = total_words / total_sentences if total_sentences > 0 else 0
    return avg_words_per_sentence

# Function to count total complex words
def count_complex_words(text):
    words = word_tokenize(text)
    complex_word_count = sum(1 for word in words if len(word) > 2)
    return complex_word_count

# Function to count total words
def count_total_words(text):
    words = word_tokenize(text)
    return len(words)

# Function to count syllables per word
def count_syllables_per_word(word):
    vowels = 'aeiouAEIOU'
    count = 0
    word = word.lower()
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count = 1
    return count

# Function to count personal pronouns
def count_personal_pronouns(text):
    personal_pronouns = ['I', 'we', 'my', 'ours', 'us']
    pronoun_count = sum(1 for word in word_tokenize(text) if word.lower() in personal_pronouns)
    return pronoun_count

# Function to calculate average word length
def calculate_avg_word_length(text):
    words = word_tokenize(text)
    total_characters = sum(len(word) for word in words)
    total_words = len(words)
    avg_word_length = total_characters / total_words if total_words > 0 else 0
    return avg_word_length

# Function to perform text analysis
def perform_text_analysis(text):
    cleaned_text = preprocess_text(text)
    positive_score, negative_score, polarity_score, subjectivity_score = calculate_sentiment_scores(text)
    avg_sentence_length = calculate_avg_sentence_length(text)
    percentage_complex_words = calculate_percentage_complex_words(text)
    fog_index = calculate_fog_index(avg_sentence_length, percentage_complex_words)
    avg_words_per_sentence = calculate_avg_words_per_sentence(text)
    complex_words_count = count_complex_words(text)
    total_word_count = count_total_words(text)
    syllables_per_word = count_syllables_per_word(text)
    personal_pronoun_count = count_personal_pronouns(text)
    avg_word_length = calculate_avg_word_length(text)

    return {
    'Positive Score': positive_score,
    'Negative Score': negative_score,
    'Polarity Score': polarity_score,
    'Subjectivity Score': subjectivity_score,
    'Avg Sentence Length': avg_sentence_length,
    'Percentage of Complex Words': percentage_complex_words,
    'Fog Index': fog_index,
    'Average number of words per sentence': avg_words_per_sentence,
    'Complex word count': complex_words_count,
    "Total word count": total_word_count,
    "Syllables per word in ": syllables_per_word,
    "Personal pronoun count": personal_pronoun_count,
    "Average word length": avg_word_length
}

######################################################################