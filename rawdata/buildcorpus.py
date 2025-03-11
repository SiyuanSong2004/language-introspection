import pandas as pd
from pathlib import Path 
import nltk
# nltk.download('wordnet')
from tqdm import tqdm
import argparse

CORPUS_PATH = Path('rawdata/BNC_COCA_en.csv')


def get_freqs(df, word):
    word_freq_row = df[df['word_form'] == word]
    return word_freq_row['log_lemma_freq'].iloc[0] if not word_freq_row.empty else 1

def get_words(df, word):
    word_freq = get_freqs(df, word)
    lemma_row = df[df['word_form'] == word]
    lemma = lemma_row['lemma'].iloc[0] if not lemma_row.empty else None

    same_freq_df = df[(df['lemma'] != lemma)].copy()
    same_freq_df['freq_diff'] = abs(same_freq_df['log_lemma_freq'] - word_freq)
    
    same_freq_df = same_freq_df.sort_values(by='freq_diff')
    
    if not same_freq_df.empty:
        same_freq_word = same_freq_df.iloc[0]['word_form']
    else:
        same_freq_word = df[(df['lemma_freq'] == 1) & (df['word_form'] != word)]['word_form'].iloc[0]
    
    freq_sfw = get_freqs(df, same_freq_word)
    
    return same_freq_word, word_freq, freq_sfw



def process_sent(sent, wordfreq_df):
    tokens = sent.split()
    continuation = tokens[-1].rstrip('".?!')
    prefix = " ".join(tokens[:-1])
    
    same_freq_word, freq, freq_sfw  = get_words(wordfreq_df, continuation)
    return prefix, continuation, same_freq_word, freq, freq_sfw


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a corpus from a given text file')
    parser.add_argument('-i', '--input', type=str, required=True, help='Name of the file to process')
    args = parser.parse_args()


    wordfreq_df = pd.read_csv(CORPUS_PATH)
    df = pd.read_csv(args.input)
    dataset = (args.input.split('/')[-1]).split('_')[0]

    results = []
    i = 0
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing sentences"):

        prefix, final_word, same_freq_word, log_freq,log_freq_alter = process_sent(row['sent'], wordfreq_df)
        results.append({
            'item_id': f'{dataset}_{i}',
            'prefix': prefix,
            'good_continuation': final_word,
            'bad_continuation': same_freq_word,
            'log_freq': log_freq,
            'log_freq_alter': log_freq_alter
        })
        i += 1


    results_df = pd.DataFrame(results)
    
    output_path = Path('testsuites', f"{dataset}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f'Corpus saved to {output_path}')
