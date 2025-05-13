from fastapi import FastAPI, Query
from pydantic import BaseModel
import uvicorn
from mainChatBot import ChatBot

app = FastAPI()
chatbot = ChatBot()

class ModelChoice(BaseModel):
    model: str

class ModelWeight(BaseModel):
    model: str
    weight_path: str

@app.get("/get_latest_news")
def get_latest_news():
    news = chatbot.get_latest_hot_news_with_summary()
    return news

@app.get("/search_news")
def search_news(query: str = Query(..., description="Search query")):
    news = chatbot.get_latest_hot_news_with_summary(userInput=query)
    return news

@app.post("/set_model")
def set_model(choice: ModelChoice):
    chatbot.set_chosen_model(choice.model)
    return {"status": "success", "chosen_model": chatbot.get_chosen_model()}

@app.post("/set_model_weight")
def set_model_weight(weight: ModelWeight):
    chatbot.set_model_weight(weight.model, weight.weight_path)
    return {"status": "success", "weight": chatbot.get_model_weight()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
