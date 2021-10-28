import requests
import pandas as pd
import nltk
from nltk.util import ngrams
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup


def web_scrape():
    URL = "http://3.85.131.173:8000/random_company"
    df = {'Name': [], 'Purpose': []}
    for i in range(50):
        r = requests.get(URL)
        soup = BeautifulSoup(r.content, 'html5lib')
        list_items = soup.find_all('li')
        Name, Purpose = None, None
        for item in list_items:
            if 'Name:' in item.text:
                Name = item.text.replace('Name: ', '')
            elif 'Purpose:' in item.text:
                Purpose = item.text.replace('Purpose: ', '')

        if Name is not None and Purpose is not None:
            df['Name'].append(Name)
            df['Purpose'].append(Purpose)
    print(df)
    df = pd.DataFrame(df)
    print(df)
    df.to_csv("web_scrape_output.csv")
    return df


def merge_data(df, Path1, Path2):
    df1 = pd.read_csv(Path1)
    df2 = pd.read_excel(Path2)
    df_list = [df, df1, df2]
    df_combined = pd.concat(df_list)
    return df_combined


def get_tokens_and_tags(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    return tokens, tags


def extract_ngrams(text, num):
    n_grams = ngrams(nltk.word_tokenize(text), num)
    return [' '.join(grams) for grams in n_grams]


def stems(text):
    return [stemmer.stem(word) for word in text]


def lemmas(text):
    return [lemmatizer.lemmatize(word) for word in text]


def sentiment_analysis(text):
    return sia.polarity_scores(' '.join(text))


def sentiment(df_combined):
    results = []
    tone = []
    for index, row in df_combined.iterrows():
        text = row["Purpose"]
        tokens, tags = get_tokens_and_tags(text)
        # list_text = extract_ngrams(tokens,tags)
        stemmed_text = stems(tokens)
        clean_text = lemmas(stemmed_text)
        print(clean_text)
        score = sentiment_analysis(clean_text)
        text_score = score['compound']
        results.append(text_score)
        if text_score > 0:
            tone.append("Positive")
        elif text_score == 0:
            tone.append("Neutral")
        elif text_score < 0:
            tone.append("Negative")
    df_combined["Score"] = pd.Series(results)
    df_combined["Tone"] = pd.Series(tone)
    return df_combined


if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    df = web_scrape()
    # df_combined = merge_data(df, "Path1", "Path2")
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    sia = SIA()
    df_final = sentiment(df)
    print(df_final)
    df_final.to_csv("nlp_scores.csv")
