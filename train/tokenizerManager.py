# tokenizer_utils.py

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import re
import string
import unicodedata
import pickle
import os

class TokenizerManager:
    def __init__(self, num_words=8000, oov_token="<OOV>"):
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        self.num_words = num_words
        self.oov_token = oov_token

    def train_tokenizer_from_csv(self, csv_path, article_col="Articles", summary_col="Summaries"):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path).dropna()
        df[article_col] = df[article_col].apply(self.clean_text)
        df[summary_col] = df[summary_col].apply(self.clean_text)
        texts = df[article_col].tolist() + df[summary_col].tolist()
        self.tokenizer.fit_on_texts(texts)
        print(f"word size: {len(self.tokenizer.word_index) + 1}")

    def save_tokenizer(self, save_path="tokenizer.pkl"):
        with open(save_path, "wb") as f:
            pickle.dump(self.tokenizer, f)
        print(f"Tokenizer is saved at: {save_path}")

    def get_tokenizer(self):
        return self.tokenizer

    @staticmethod
    def load_tokenizer(load_path="tokenizer.pkl"):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Tokenizer file not found: {load_path}")
        with open(load_path, "rb") as f:
            tokenizer = pickle.load(f)
        return tokenizer
    
    @staticmethod
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    
    @staticmethod
    def clean_text(text):
        text = TokenizerManager.unicode_to_ascii(text.lower().strip())
        # convert ... and .. to <title_end> and <p>
        text = re.sub(r"\.\.\.", " <p> ", text)
        text = re.sub(r"\.\.", " <title_end> ", text)
        text = re.sub(r"\.", " ", text)      # 单个句号
        text = re.sub(r",", " ", text)       # 逗号
        
        # Abbreviation Restoration & Stem Preservation
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)

        # delete most punctuation marks
        text = re.sub(r"[-\"#/@;:{}`+=~|]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text =  "<sos> " +  text + " <eos>"
        return text
    
    @staticmethod
    def clean_text_for_inference(text):
        text = TokenizerManager.unicode_to_ascii(text.lower().strip())
        # convert ... and .. to <title_end> and <p>
        # however, we can not use "<>" since it will be removed by tokenizer in default
        text = re.sub(r"\.\.\.", " p ", text)
        text = re.sub(r"\.\.", " title_end ", text)
        text = re.sub(r"\.", " ", text)      # 单个句号
        text = re.sub(r",", " ", text)       # 逗号
        
        # Abbreviation Restoration & Stem Preservation
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)

        # delete most punctuation marks
        text = re.sub(r"[-\"#/@;:{}`+=~|]", "", text)

        # Multiple spaces merge
        text = re.sub(r"\s+", " ", text).strip()
        text =  "sos " +  text + " eos"
        return text

    
