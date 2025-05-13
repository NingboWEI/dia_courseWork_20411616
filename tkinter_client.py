import tkinter as tk
import requests

API_URL = "http://127.0.0.1:8000"

def get_latest_news():
    response = requests.get(f"{API_URL}/get_latest_news")
    news = response.json()
    display_news(news)

def search_news():
    query = entry_search.get()
    response = requests.get(f"{API_URL}/search_news", params={"query": query})
    news = response.json()
    display_news(news)

def set_model():
    model = combo_model.get()
    requests.post(f"{API_URL}/set_model", json={"model": model})

def set_model_weight():
    model = combo_model.get()
    weight_path = entry_weight.get()
    requests.post(f"{API_URL}/set_model_weight", json={"model": model, "weight_path": weight_path})

def display_news(news):
    text_display.delete(1.0, tk.END)
    for item in news:
        text_display.insert(tk.END, f"Title: {item['title']}\n",  ("bold",))
        text_display.insert(tk.END, f"Url: {item['url']}\n")
        text_display.insert(tk.END, f"Summary: {item['summary']}\n{'-'*40}\n")

app = tk.Tk()
app.title("News Summarizer")
app.geometry("700x800")

frame_top = tk.Frame(app, padx=10, pady=10)
frame_top.pack(fill=tk.X)

label_search = tk.Label(frame_top, text="Search News:")
label_search.pack(side=tk.LEFT)

entry_search = tk.Entry(frame_top, width=40)
entry_search.pack(side=tk.LEFT)

button_search = tk.Button(frame_top, text="Search", command=search_news)
button_search.pack(side=tk.LEFT)

frame_buttons = tk.Frame(app, padx=10, pady=10)
frame_buttons.pack()

button_latest = tk.Button(frame_buttons, text="Latest News", command=get_latest_news)
button_latest.grid(row=0, column=0, padx=5)

combo_model = tk.StringVar()
combo_model.set("LSTM")
model_options = tk.OptionMenu(frame_buttons, combo_model, "LSTM", "GRU", "T5", "textRank")
model_options.grid(row=0, column=1, padx=5)

entry_weight = tk.Entry(frame_buttons, width=20)
entry_weight.grid(row=0, column=2, padx=5)

button_set_model = tk.Button(frame_buttons, text="Set Model", command=set_model)
button_set_model.grid(row=0, column=3, padx=5)

button_set_weight = tk.Button(frame_buttons, text="Set Model Weight", command=set_model_weight)
button_set_weight.grid(row=0, column=4, padx=5)

text_display = tk.Text(app, height=55, width=100)
text_display.tag_configure("bold", font=("Arial", 10, "bold"))
text_display.pack(padx=10, pady=10)

app.mainloop()
