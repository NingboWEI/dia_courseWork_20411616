# Dia_Ningbo -- NewsCompanion – Your Personalized Daily News Assistant

use `main.ipynb` to run the code

## Enviornment suggest
- this system requires `tensorflow` and `pytorch`
- A `cuda` device with high video memory can help quick inference
- you might need to download `nltk-data` if you want to try textRank. You can achieve this by pip install

## key point of this system
- doing text summarization with different strategies on BBC news
- using language methods:
    - text extraction method(TextRank)
    - local trained generative method(LSTM and GRU)
    - pre-trained large language model with fine-tuning(T5-small)

## future work
- better approaches in cutomized tokenizer, including using sub-word technick
- better configuration for local trained generative method
- applying more powerful hardware to support bigger data set training
- achieve RAG functions(user requirement)











