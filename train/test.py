import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenizerManager import TokenizerManager

print("start to do tokenizer")
TOKENIZER_MANAGER = TokenizerManager()
TOKENIZER_MANAGER.train_tokenizer_from_csv("../data/bbc-news-summary.csv")
TOKENIZER_MANAGER.save_tokenizer("../data/tokenizer.pkl")
print("tokenizer done")