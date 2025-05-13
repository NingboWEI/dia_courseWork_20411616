from model.GRU_seq2seq import GRUAttentionModel
from train.tokenizerManager import TokenizerManager
import tensorflow as tf

import unicodedata
from collections import Counter
import os
import sys
import re

class GRUpakage:
    def __init__(self, modelPath='train/checkpoint/GRU_seq2seq_bbc_500_fixed.h5'):
        self.tokenizer = TokenizerManager.load_tokenizer("data/tokenizer_fixed.pkl")
        embed_dim = 256
        hidden_dim = 512
        num_heads = 8
        dropout_rate = 0.3
        vocab_size = 10000 + 1
        self.model = GRUAttentionModel(
            vocab_size=vocab_size, 
            embed_dim=embed_dim, 
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout_rate=dropout_rate
        )
        self.modelPath = modelPath
        self.loadWeight()


    def loadWeight(self, modelPath=None):
        if modelPath==None:
            modelPath = self.modelPath
        dummy_input = tf.zeros((1, 600), dtype=tf.int32)  # 1是批次大小，600是你的输入序列长度
        dummy_decoder_input = tf.zeros((1, 160), dtype=tf.int32)  # 160是你的输出序列长度
        self.model(dummy_input, dummy_decoder_input)  # 构建模型
        self.model.load_weights(modelPath)

    def response(self, input_text):
        top_oov_words = self.find_high_frequency_oov_words(input_text, self.tokenizer, top_n=10)
        # print("\nTop 10 High-Frequency OOV Words:", top_oov_words)
        if len(top_oov_words)>0:
            mostWord, frq = top_oov_words[0]
        else:
            mostWord = ""
        output_text = self.infer(self.model, self.tokenizer, input_text)
        output_text = output_text.replace("<OOV>", mostWord)
        return output_text

    def clean_text_for_inference(self, text):
        def unicode_to_ascii(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
        text = unicode_to_ascii(text.lower().strip())
        # convert ... and .. to <title_end> and <p>
        # however, we can not use "<>" since it will be removed by tokenizer in default
        text = re.sub(r"\.\.\.", " p ", text)
        text = re.sub(r"\.\.", " <OOV> ", text)
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
        text = text.replace("<oov>", "<OOV>")
        return text

    def find_high_frequency_oov_words(self, input_text, tokenizer, top_n=10):
        # 1. 清理文本，去除标点符号
        clean_text = self.clean_text_for_inference(input_text)
        words = clean_text.split()

        # 2. 获取 Tokenizer 词汇表
        vocab = set(tokenizer.word_index.keys())

        # 3. 统计未见词（OOV）
        oov_words = [word for word in words if word not in vocab]
        oov_counter = Counter(oov_words)

        # 4. 返回按频率排序的前 N 个未见词
        top_oov_words = oov_counter.most_common(top_n)
        return top_oov_words

    def infer(self, model, tokenizer, input_text, max_length=160):
        # 清理输入文本
        input_text = self.clean_text_for_inference(input_text)
        input_text = input_text.replace("<oov>", "<OOV>")
        input_seq = tokenizer.texts_to_sequences([input_text])
        input_seq = tf.keras.preprocessing.sequence.pad_sequences(input_seq, maxlen=605, padding="post")

        # 初始化解码器输入 (<sos>)
        decoder_input = tf.constant([[tokenizer.word_index['sos']]])

        result = []
        for _ in range(max_length):
            # 模型推理
            predictions = model(
                tf.convert_to_tensor(input_seq, dtype=tf.int32),
                tf.convert_to_tensor(decoder_input, dtype=tf.int32)
            )
            predicted_id = tf.argmax(predictions[0, -1, :]).numpy()

            # 如果预测是 <eos>，则结束
            if predicted_id == tokenizer.word_index['eos']:
                break

            # 将预测词加入结果
            result.append(predicted_id)
            decoder_input = tf.concat([decoder_input, tf.constant([[predicted_id]])], axis=-1)

        # 将 ID 转回文本
        result_text = tokenizer.sequences_to_texts([result])[0]
        return result_text[4:]