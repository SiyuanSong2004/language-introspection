import random
from nltk.corpus import wordnet
import pandas as pd
import random
from tqdm import tqdm
from wikipedia.exceptions import DisambiguationError, PageError

random.seed(46)
synsets = list(wordnet.all_synsets())

data = []

vocab_df = pd.read_csv("rawdata/BNC_COCA_en.csv")
top_k_lists = ['1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k']
bnc_top8k = vocab_df[vocab_df['List'].isin(top_k_lists)]
vocab = set(bnc_top8k['word_form'])

def generate_random_sentence(min_length=8, max_length=20, final_words=vocab):
    sentence_length = random.randint(min_length, max_length)
    words = []
    for _ in range(sentence_length-1):
        random_synset = random.choice(synsets)
        random_word = random_synset.lemma_names()[0]
        words.append(random_word.replace('_', ' ')) 
    final_word = random.choice(list(final_words))
    words.append(final_word)

    return str(" ".join(words))

with tqdm(total=1000) as pbar:
    while len(data) < 1000:
        try:
            sent = generate_random_sentence()
            data.append({"sent": sent})
            pbar.update(1)
        except Exception as e:
            print(f"Error: {e}")
            continue

df = pd.DataFrame(data)
df.to_csv("rawdata/randomseq/randomseq_random_samples.csv", index=False)


