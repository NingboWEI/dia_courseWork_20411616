# read bbc-news dataset and prepare it for training

import pandas as pd
import torch
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tokenizerManager import TokenizerManager

from torch.utils.data import Dataset
import unicodedata
import os
import sys
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, src_max_len=600, trg_max_len=160):
        self.texts = df['Articles'].tolist()
        self.summaries = df['Summaries'].tolist()
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len

        self.src_seqs = pad_sequences(tokenizer.texts_to_sequences(self.texts),
                                      maxlen=self.src_max_len, padding="post", truncating="post")

        self.trg_seqs = pad_sequences(tokenizer.texts_to_sequences(self.summaries),
                                      maxlen=self.trg_max_len, padding="post", truncating="post")

    def __len__(self):
        return len(self.src_seqs)

    def __getitem__(self, idx):
        src_seq = self.src_seqs[idx]
        trg_seq = self.trg_seqs[idx]

        return torch.tensor(src_seq), torch.tensor(trg_seq)

def get_dataloaders(csv_path, tokenizer, loadDataSize, batch_size=16):
    df = pd.read_csv(csv_path).dropna()
    print(df.columns)
    per_class_count = int(loadDataSize / 5)
    selected_dfs = []

    for category in df['File_path'].unique():
        category_df = df[df['File_path'] == category].head(per_class_count)
        selected_dfs.append(category_df)

    selected_df = pd.concat(selected_dfs).sample(frac=1).reset_index(drop=True)  # shuffle the selected data

    # 清理文本并添加 <sos> 和 <eos>
    selected_df['Articles'] = selected_df['Articles'].apply(TokenizerManager.clean_text_for_inference)
    selected_df['Summaries'] = selected_df['Summaries'].apply(TokenizerManager.clean_text_for_inference)


    train_size = int(0.9 * loadDataSize)
    train_df = selected_df.iloc[:train_size]
    val_df = selected_df.iloc[train_size: loadDataSize]

    train_dataset = NewsDataset(train_df, tokenizer)
    val_dataset = NewsDataset(val_df, tokenizer)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader

def clean_text_for_inference(text):
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    text = unicode_to_ascii(text.lower().strip())
    # convert ... and .. to <title_end> and <p>
    # however, we can not use "<>" since it will be removed by tokenizer in default
    text = re.sub(r"\.\.\.", " p ", text)
    text = re.sub(r"\.\.", " title_end ", text)
    
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
    text =  " sos " +  text + " eos "
    return text