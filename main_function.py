import re
import string
from wordcloud import STOPWORDS
from nltk.tokenize import TweetTokenizer
import nltk
from gensim.models import Word2Vec
import numpy as np

class X_clean():
    def __init__(self, x):
        self.x = x 
    
    def data_clean(self):
        x = re.sub(r'https?://\S+|www\.\S+',"",self.x)
        x = re.sub(r'<.*?>',"",x)
        x = re.sub(r"\x89ÛÒ", "", x)
        emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
        x = emoji_pattern.sub(r'', x)
        x = x.replace("%20", "_")
        x = re.sub(r"@[^\s]+[\s]?", "", x)
        table = str.maketrans('', '', string.punctuation)
        x = x.translate(table)
        return x
    
    def remove_stopwords(self):
        x = self.data_clean()  # Use the output of data_clean() method
        x = ' '.join([word for word in x.split() if word not in STOPWORDS])
        return x
    
    def tokenize_and_lemmatize(self):
        x = self.remove_stopwords()  # Use the output of remove_stopwords() method
        tokenizer = TweetTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        x = x.lower()
        tokens = tokenizer.tokenize(x)
        x = [lemmatizer.lemmatize(token) for token in tokens]
        return x
    
    def get_embedding(self):
        x = self.tokenize_and_lemmatize()  # Use the output of tokenize_and_lemmatize() method
        pretrained_model_path = "word2vec_model"
        model_vec = Word2Vec.load(pretrained_model_path)
        embeddings = []
        for token in x:
            try:
                # Try to get embedding for the token directly
                embedding = model_vec.wv[token]
                embeddings.append(embedding)
            except KeyError:
                # If token is OOV, skip it
                pass
        return embeddings

    def average_embed(self):
        embeddings = self.get_embedding()  # Use the output of get_embedding() method
        vector_size = 256
        embedding = np.zeros(vector_size)
        wcount = 1
        for x in embeddings:
            embedding += x
            wcount += 1
        embedding = embedding / wcount
        return embedding
