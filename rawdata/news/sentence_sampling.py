import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import re

INCLUDE_TITLE = True
data = []
item_num = 0
english_pattern = re.compile(r'^[A-Za-z0-9\s.,!?\'"-]+$')
vocab_df = pd.read_csv("rawdata/BNC_COCA_en.csv")
top_k_lists = ['1k', '2k', '3k', '4k', '5k', '6k', '7k', '8k']
bnc_top8k = vocab_df[vocab_df['List'].isin(top_k_lists)]
vocab = set(bnc_top8k['word_form'])
df = pd.read_csv("rawdata/news/cleaned.csv")
print(df.head())
print("Number of items:", len(df.index))
df.drop_duplicates(inplace=True) # in case one news article is double counted
print("Number of items after dropping duplicates:", len(df.index)) 
    
for _, row in tqdm(list(df.iterrows()), total=len(df.index)):
    title = row.title
    text = row.raw_text

    try:
        sentences = sent_tokenize(text)
    except:
        print("Skipping:", title)
        continue
        
    for sent in sentences:
        sent = sent.replace('\n', ' ').replace('\r', '').strip()  
        for eos in [".", "?", "!"]:
                if sent.endswith(eos):
                    sent = sent[:-1]
        sent = re.sub(r'[^\x20-\x7E]+', ' ', sent)  
        tokens = sent.split()


        if not (english_pattern.match(sent) and 8<=len(tokens)<=25 and '\n' not in sent and tokens[-1] in vocab):
            continue  

        # if sentence has been processed
        if sent in [d['sent'] for d in data]:
            continue

        data.append(dict(
            title=title,
            sent=sent,
            category=row.category,
            published_date_gmt=row.published_date_gmt,
            download_date=row.download_date
        ))

        # Update counter.
        item_num += 1

print("Final number of items:", item_num)
corpus = pd.DataFrame(data)
# sample 1000 sentences
corpus = corpus.sample(1000)
print(corpus.head())
corpus.to_csv("rawdata/news/news_random_samples.csv", index=False)