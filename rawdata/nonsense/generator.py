# label sentence with universal POS tag: https://www.nltk.org/book/ch05.html
# then replace with same corpus frequency
import pandas as pd
import nltk
import random
from nltk.corpus import wordnet as wn
import tqdm 

# download if needed
#nltk.download('averaged_perceptron_tagger')
#nltk.download('punkt')
#nltk.download('universal_tagset')


corpus = pd.read_csv('rawdata/BNC_COCA_en_pos.csv')
# drop UNK
corpus = corpus[corpus['POS'] != 'UNK']
corpus_8k = corpus[corpus['List'].isin(['1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k'])]


def get_random_word(pos_tag, final_word=False):
    if final_word:
        words = corpus_8k
    else:
        words = corpus
    pos_corpus = words[words['POS'].apply(lambda pos_list: pos_tag in eval(pos_list))]
    if pos_corpus.empty:
        return None
    row = pos_corpus.sample(1)
    return row['lemma'].iloc[0]



def replace_words_with_same_pos(sentence):
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words, tagset='universal')
    
    new_sentence = []
    r = 0
    nr = 0
    # process until the final word
    for word, pos in pos_tags[:-2]:
        new_word = get_random_word(pos)
        if new_word:
            new_sentence.append(new_word)
            r += 1
        else:
            new_sentence.append(word) 
            nr += 1
    # add the final word
    final_word = get_random_word(pos_tags[-1][1], final_word=True)
    if final_word:
        new_sentence.append(final_word)
        r += 1
    else:
        new_sentence.append(pos_tags[-1][0])
        nr += 1
    
    # convert to string
    new_sentence = [str(word) for word in new_sentence]
    return ' '.join(new_sentence), r, nr


if __name__ == '__main__':
    random.seed(46)
    original_df = pd.read_csv('rawdata/news/news_random_samples.csv')
    for i, row in tqdm.tqdm(original_df.iterrows(), total=len(original_df)):
        sent = row['sent']
        new_sent, r, nr = replace_words_with_same_pos(sent)
        original_df.at[i, 'sent'] = new_sent
        original_df.at[i, 'replaced'] = r
        original_df.at[i, 'not_replaced'] = nr
    original_df.to_csv('rawdata/jabberwocky/jabberwocky_random_samples_pos.csv', index=False)
    

