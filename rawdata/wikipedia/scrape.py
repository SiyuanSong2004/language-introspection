import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from newspaper import Article
from tqdm import tqdm

url = "https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/3"

'''
Get the URLs of the vital articles from Wikipedia (Level 3, n=1000)
'''

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")


data = []
for a in soup.select("a[href^='/wiki/']"): 
    href = a['href']
    
    if '.svg' in href or href.count('/') > 2:
        continue
    
    title = a.text.strip()
    full_url = f"https://en.wikipedia.org{href}" 
    data.append({"title": title, "url": full_url})


df = pd.DataFrame(data)
first_index = df[df['title'] == 'Hammurabi'].index
end_index = df[df['title'] == 'Statistics'].index
if not first_index.empty and not end_index.empty:
    df = df.loc[first_index[0]:end_index[0]]


en = re.compile(r'^[A-Za-z0-9\s]+$')
df = df[df['title'].apply(lambda x: bool(en.match(x)))].reset_index(drop=True)

'''
Scrape the text of the articles
'''
scraped_data = []  

for _, row in tqdm(df.iterrows(), total=len(df.index)):
    url = row["url"]
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        scraped_data.append({
            "title": article.title,
            "url": url,
            "raw_text": article.text
        })
        
    except Exception as e:
        print(f"Encountered error for {url}: {e}")
        continue


texts = pd.DataFrame(scraped_data)
print(texts.head())
print("Number of texts:", len(texts.index))


texts.to_csv("cleaned.csv", index=False)
