from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
import unicodedata

# 有待完善summary的处理

class T5Small:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def generate(self, input_text):
        inputs_token = self.tokenizer.encode("summarize: " +input_text)
        # print(f"inputs_token length: {len(inputs_token)}")
        if len(inputs_token) > 510:
            # print("input text is too long, using chunk method")
            summary = self.summarize_text_withChunk(input_text, self.tokenizer, self.model)
        else:
            # print("input text is short, using normal method")
            final_input_tokens = self.tokenizer.encode("summarize: " +input_text, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(final_input_tokens, max_length=min(170, len(inputs_token)//2), min_length=100, length_penalty=1.5, num_beams=4, early_stopping=True)
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return summary
    
    def chunk_text(self, text, tokenizer, max_tokens=400):
        tokens = tokenizer.encode(text)
        chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

    def summarize_text_withChunk(self, text, tokenizer, model):
        total_tokens = len(tokenizer.encode(text))
        chunks = self.chunk_text(text, tokenizer)
        partial_summaries = []
        
        for chunk in chunks:
            input_text = "summarize: " + chunk
            inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = model.generate(inputs, max_length=100, min_length=55, num_beams=4, early_stopping=True)
            partial_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            partial_summaries.append(partial_summary)
            # print(f"Partial summary: {partial_summary}")

        # gether all partial summaries and summarize them again
        combined_summary = " ".join(partial_summaries)
        final_input = "summarize: " + combined_summary
        final_inputs = tokenizer.encode(final_input, return_tensors="pt", truncation=True, max_length=512)
        final_summary_ids = model.generate(final_inputs, max_length=min(170, total_tokens//2), min_length=100, num_beams=4, early_stopping=True)
        final_summary = tokenizer.decode(final_summary_ids[0], skip_special_tokens=True)
        # print(f"Final summary: {final_summary}")

        return final_summary
    

    
    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def clean_text(self, text):
        text = self.unicode_to_ascii(text.lower().strip())
        # convert ... and .. to <title_end> and <p>
        text = re.sub(r"\.\.\.", ". ", text)
        text = re.sub(r"\.\.", ". ", text)
        
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
        return text