# add POS tags (or 'UNK' if not found) to the BNC_COCA_en.csv file and save it as BNC_COCA_en_pos.csv
# use brown corpus to get POS tags, refer to https://stackoverflow.com/questions/44382254/nltk-single-word-part-of-speech-tagging
import pandas as pd
import nltk

all_data = pd.read_csv('BNC_COCA_en.csv')
all_lemma = all_data.drop_duplicates(subset='lemma')
all_lemma = all_lemma[['List', 'lemma', 'log_lemma_freq']]


wordtags = nltk.ConditionalFreqDist((w.lower(), t) 
        for w, t in nltk.corpus.brown.tagged_words(tagset="universal"))


def get_pos(word):
    word = str(word)
    if word.lower() not in wordtags:
        return 'UNK'
    return list(wordtags[word.lower()])


all_lemma['POS'] = all_lemma['lemma'].apply(get_pos)
all_lemma.to_csv('BNC_COCA_en_pos.csv', index=False)