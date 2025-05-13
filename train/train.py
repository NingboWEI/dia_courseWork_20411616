import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from model.model import Encoder, Decoder, Seq2Seq, Attention
from data.utils import get_dataloaders
from data.utils_bbc import get_dataloaders as get_dataloaders_bbc
import pandas as pd

# 读取所有文本来训练 tokenizer（记得提前加载数据）
all_data = pd.read_csv("../data/bbc-news-summary.csv").dropna()
texts = all_data["Articles"].tolist() + all_data["Summaries"].tolist()

# 初始化 tokenizer
TOKENIZER = Tokenizer(num_words=8000, oov_token="<OOV>")
TOKENIZER.fit_on_texts(texts)

# 用于 pad 的 token_id 设置
PAD_TOKEN_ID = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained("t5-small")  # 使用 T5 的 tokenizer
LOADDATASIZE = 500  # row of data
BATCH_SIZE = 8  # batch size

def train_model():
    train_loss_history = []
    val_loss_history = []
    # train_loader, val_loader = get_dataloaders("../data/NewsSummary.parquet", TOKENIZER, LOADDATASIZE, BATCH_SIZE)
    train_loader, val_loader = get_dataloaders_bbc("../data/bbc-news-summary.csv", TOKENIZER, LOADDATASIZE, BATCH_SIZE)

    INPUT_DIM = len(TOKENIZER)
    OUTPUT_DIM = len(TOKENIZER)
    EMB_DIM = 256  # embedding dimension for both encoder and decoder
    HID_DIM = 512  # hidden dimension for LSTM
    N_LAYERS = 2

    # 1. Attention
    attn = Attention(enc_hid_dim=HID_DIM, dec_hid_dim=HID_DIM)

    # 2. Encoder 和 Decoder（Decoder 加入 attn）
    enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, N_LAYERS) # 怎么设置
    dec = Decoder(OUTPUT_DIM, EMB_DIM, enc_hid_dim=HID_DIM, dec_hid_dim=HID_DIM,
                attention=attn, num_layers=N_LAYERS)  # 怎么设置

    # 3. Seq2Seq
    model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=TOKENIZER.pad_token_id)   # use cross entropy loss for seq2seq model

    for epoch in range(300):
        # ✅ Training
        model.train()
        total_train_loss = 0
        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # ✅ Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, trg in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                src, trg = src.to(DEVICE), trg.to(DEVICE)
                output = model(src, trg)
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

   # ✅ 保存模型
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(model.state_dict(), "checkpoint/seq2seq_bbc_500.pt")
    print("✅ 模型训练完成并已保存。")

    # ✅ 绘图：训练集 & 验证集 Loss 曲线
    print("==============================")
    plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_loss_history)+1), val_loss_history, label='Val Loss', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("checkpoint/loss_curve_bbc_500.png")
    # plt.show()
    print("✅ Loss 曲线图已保存。")


if __name__ == "__main__":
    train_model()
