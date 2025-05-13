# back-end code for the chatbot
import sys
import os
import torch
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model.LSTM_seq2seq import Encoder, Decoder, Attention, Seq2Seq
from model.T5_small import T5Small
from model.textRank import TextRank
from model.GRUpackage import GRUpakage
from train.tokenizerManager import TokenizerManager
from newsExtract.newsCrawler import NewsCrawler

PATH_LSTM = "train/checkpoint/seq2seq_bbc_500_fixed.pt"
PATH_GRU = "train/checkpoint/GRU_seq2seq_bbc_500_fixed.h5"

class ChatBot:
    def __init__(self, path_to_LSTM_model=PATH_LSTM, path_to_GRU_model=PATH_GRU):
        self.newsExtracter = NewsCrawler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_LSTM = TokenizerManager.load_tokenizer("data/tokenizer_fixed.pkl")
        self.path_to_LSTM_model = path_to_LSTM_model
        self.path_to_GRU_model = path_to_GRU_model
        self.model_LSTM = self.get_LSTM_model(self.path_to_LSTM_model)
        self.model_T5 = T5Small()
        self.model_textRank = TextRank()
        self.model_GRU = GRUpakage()
        self.chosen_model = "LSTM"  # LSTM; T5; textRank; GRU

        self.set_model_weight("GRU", path_to_GRU_model)
        self.set_model_weight("LSTM", path_to_LSTM_model)

        # TextRank.environment_check()  # Check if the environment is set up correctly for TextRank

    def get_latest_hot_news(self, limit=3):
        """
        get the latest hot news from the web
        """
        return self.newsExtracter.extract_latest_hot_news(limit=limit)
    
    def get_latest_hot_news_with_summary(self, userInput=None, limit=3):
        """
        get the latest hot news from the web and summarize it
        """
        if userInput==None:
            news = self.newsExtracter.extract_latest_hot_news(limit=limit)
        else:
            news = self.newsExtracter.extract_search_news(quary=userInput, limit=limit)
        len_news = len(news)
        for i in range(len_news):
            title = news[i]["title"]
            article = news[i]["content"]
            summary = self.news_summary(title, article)
            news[i]["summary"] = summary
        return news
    
    def news_summary(self, inputTitle, inputtext):
        if self.chosen_model == "LSTM":
            return self.predict_LSTM(inputTitle, inputtext)
        elif self.chosen_model == "GRU":
            return self.predict_GRU(inputTitle, inputtext)
        elif self.chosen_model == "T5":
            return self.model_T5.generate(inputtext)
        elif self.chosen_model == "textRank":
            return self.model_textRank.generate_summary(inputtext, ratio=0.3)
        else:
            return "UNKONWN MODEL"
        
    def set_chosen_model(self, model_name):
        """
        model_name: LSTM; T5
        """
        if model_name == "LSTM":
            self.chosen_model = "LSTM"
        elif model_name == "T5":
            self.chosen_model = "T5"
        elif model_name == "textRank":
            self.chosen_model = "textRank"
        elif model_name == "GRU":
            self.chosen_model = "GRU"
        else:
            print("Model not found, using LSTM as default")
            self.chosen_model = "LSTM"

    def get_chosen_model(self):
        return self.chosen_model
    
    def set_model_weight(self, model_name, path_to_model):
        if model_name == "LSTM":
            self.path_to_LSTM_model = path_to_model
            self.model_LSTM = self.get_LSTM_model(self.path_to_LSTM_model)
        elif model_name == "GRU":
            self.path_to_GRU_model = path_to_model
            self.model_GRU.loadWeight(self.path_to_GRU_model)
        elif model_name == "T5" or model_name == "textRank":
            pass # T5 model is loaded in the constructor
        else:
            print("Model not found, not weight is set")

    def get_model_weight(self):
        if self.chosen_model == "LSTM":
            return self.path_to_LSTM_model
        elif self.chosen_model == "GRU":
            return self.path_to_GRU_model
        elif self.chosen_model == "T5":
            return "T5-small"
        elif self.chosen_model == "textRank":
            return "textRank"
        else:
            return "UNKONWN"
    
    # ================== LSTM model ===================

    def get_LSTM_model(self, path_to_LSTM_model=PATH_LSTM):
        INPUT_DIM = self.tokenizer_LSTM.num_words + 1
        OUTPUT_DIM = self.tokenizer_LSTM.num_words + 1
        EMB_DIM = 256
        HID_DIM = 512
        NUM_LAYERS = 2
        attn = Attention(HID_DIM, HID_DIM)
        enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, num_layers=NUM_LAYERS)
        dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, HID_DIM, attn, num_layers=NUM_LAYERS)
        model = Seq2Seq(enc, dec, self.device).to(self.device)
        model.load_state_dict(torch.load(path_to_LSTM_model, map_location=self.device))
        return model

    def preprocess_input_LSTM(self, sentence, tokenizer, max_len=600):
        sentence = TokenizerManager.clean_text_for_inference(sentence)
        # print("cleaned sentence:", sentence)
        sequence = tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post')
        return torch.LongTensor(padded)
    
    def predict_LSTM(self, title, sentence, max_len=160):
        word2idx = self.tokenizer_LSTM.word_index
        idx2word = {v: k for k, v in word2idx.items()}

        sentence = title + " <OOV> " + sentence
        src_tensor = self.preprocess_input_LSTM(sentence, self.tokenizer_LSTM, max_len).to(self.device)

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.model_LSTM.encoder(src_tensor)
            input_token = torch.LongTensor([[self.tokenizer_LSTM.word_index["sos"]]]).to(self.device)
            result = []

            for _ in range(max_len):
                output, hidden, cell = self.model_LSTM.decoder(input_token.squeeze(1), hidden, cell, encoder_outputs)
                top1 = output.argmax(1)
                word = idx2word.get(top1.item(), "<unk>")
                if word == "eos":
                    break
                result.append(word)
                input_token = top1.unsqueeze(1)

        return " ".join(result)
    
    # ================== TextRank model ===================
    def textRank_environment_check(self):
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK Punkt data is available.")
        except Exception as e:
            print("❌ NLTK Punkt data is not available. Please download it using nltk.download('punkt')")
            raise e

    # ================== GRU model ===================
    def predict_GRU(self, title, sentence):
        sentence = title + " <OOV> " + sentence
        summary = self.model_GRU.response(sentence)
        return summary

