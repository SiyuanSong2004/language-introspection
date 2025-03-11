# Dataset Construction

## Experiment 1: Acceptability Judgment
Get minimal pairs for LM acceptability judgment.

`blimpli/`:

- `BLiMP_pairs.csv`: Full BLiMP dataset from [Warstadt et al. (2020)](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00321/96452/BLiMP-The-Benchmark-of-Linguistic-Minimal-Pairs).

- `LI_Large_pairs.csv`: Sentence pairs for accetability judgments from [Sprouse et al.(2013)](https://www.sciencedirect.com/science/article/pii/S0024384113001526?casa_token=En7lS3L9HD8AAAAA:v3Vl5HIRv9HGmbXx7uMqBiMjs7FdvDJ6lwtTsYOAekvhwnCdr4CsTClZG6HQ2UyyAEXWOTGBHSs) and [Mahowald et al. (2016)](https://muse.jhu.edu/pub/24/article/629764/summary?casa_token=TuNaYv4FF-EAAAAA:TE7EgNGyf1pTe9kTgVW1uReSIo5f9k7HS7L4ZAdM31Llxyreh9bS9n_6MNeMLvGKyYazTSH4PAs). This dataset is used in Hu et al. (2025).

- `datasampler.ipynb`: Randomly sample 10 pairs from each paradigm in `BLiMP`, and select 'minimal pairs' where two sentences only differ in one word in `LI_Large`, merge and save as `/testsuites/BLIMPLI.csv`

## Experiment 2: Word Prediction


We get BNC_COCA word frequency list `BNC_COCA_en.csv` from [BNC/COCA list](https://www.eapfoundation.com/vocab/general/bnccoca/). Then, we use `process_corpus_pos` to get the part-of-speech information from Brown Corpus with `nltk`. The complete word list with part-of-speech information is `BNC_COCA_en_pos.csv`.

### Sentence Sampling
We sampled 4*1000 sentences. We attempted to keep the sentence length within 8 to 25 words and the final word in 'top 8k' BNC-COCA lists:

- Wikipedia
  - Run `scrape.py` to scrape sentences from Wikipedia Vital Articles (Level 3) and save to a file `cleand.csv` with three columns: `title`, `url` and `raw_text`.
  - Run `sentence_sampling.py` to get 1000 random sentences from `cleaned.csv`; sampled sentences are in `wikipedia_random_samples.csv`

- News
  - Download news data from [NewsData](https://newsdata.io/search-news); for this study, I downloaded **English** news with topics of *Business, Education, Entertainment, Health, Politics, Science, Sports and Tourism* in *the United States*. This was conducted in various days between 10/26/2024 and 10/31/2024.
  - Use `scrape.py` to scrape full articles from the links and save to `cleaned.csv`.
  - Use `sentence_sampling.py` to sample random sentences(`news_random_samples.csv`) from `cleaned.csv`.



- Nonsense (or `jabberwocky` in this repo)
Use `nonsense/generator.py` to process: for each sentence in [news](rawdata/news/news_random_samples.csv) dataset, replace each word in the sentence with a word of same POS, using the POS labels in [labelled BNC/COCA corpus](rawdata/BNC_COCA_en_pos.csv).



- Randomseq
Use `randomseq/generator.py` to generate sequences with randomly picked words.


Note: I referred to [Hu & Levy (2023)](https://github.com/jennhu/metalinguistic-prompting/tree/89d1b526d0ef0b955b886a9cee50883eeaee8e76/datasets) when sampling `wikipedia` and `news` data.

### Testsuite Generation

Use `buildcorpus.py` to generate testsuites by getting a alternative word with same lemma frequency with the last word in each sentence. The lemma frequency information is in the BNC/COCA lists. 

Run the following command:

`python rawdata/buildcorpus.py -i rawdata/wikipedia/wikipedia_random_samples.csv`

and the generated testsuite will be saved to `/testsuites/wikipedia.csv`




