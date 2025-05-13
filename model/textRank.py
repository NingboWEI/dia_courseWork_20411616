import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

class TextRank:
    def __init__(self):
        self.text = ""

    def read_article(self, text):
        sentences = sent_tokenize(text)
        return [word_tokenize(sentence) for sentence in sentences]

    def sentence_similarity(self, sent1, sent2, stop_words=None):
        if stop_words is None:
            stop_words = set(stopwords.words('english'))
            
        sent1 = [w.lower() for w in sent1 if w not in stop_words]
        sent2 = [w.lower() for w in sent2 if w not in stop_words]
        
        all_words = list(set(sent1 + sent2))
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        for word in sent1:
            vector1[all_words.index(word)] += 1

        for word in sent2:
            vector2[all_words.index(word)] += 1

        cosine_sim = 1 - cosine_distance(vector1, vector2)
        return cosine_sim

    def build_similarity_matrix(self, sentences, stop_words):
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 != idx2:
                    similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

        return similarity_matrix

    def generate_summary(self, text, ratio=0.2):
        stop_words = set(stopwords.words('english'))
        sentences = sent_tokenize(text)  # 使用原始句子
        tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
        total_sentences = len(sentences)
        
        # 计算摘要句子的数量，至少为1句
        summary_count = max(1, int(total_sentences * ratio))

        similarity_matrix = self.build_similarity_matrix(tokenized_sentences, stop_words)
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)

        # 获取每个句子的分数
        ranked_sentences = sorted(((scores[i], i) for i in range(len(sentences))), reverse=True)

        # 根据评分挑选前n句，并按照原文顺序排序
        selected_indices = sorted([ranked_sentences[i][1] for i in range(min(summary_count, len(ranked_sentences)))])
        summary = ' '.join([sentences[i] for i in selected_indices])

        return summary
    
    def environment_check():
        """
        Check if the environment is set up correctly for TextRank
        """
        try:
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK Punkt data is available.")
        except Exception as e:
            print(f"❌ Error setting up NLTK environment: {e}")
            print(f"Downloading NLTK data...")
            nltk.download('punkt')
            nltk.download('stopwords')
            print("✅ NLTK data downloaded successfully, suggest to restart the program.")
            