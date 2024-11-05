import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from fastapi import FastAPI, HTTPException
from starlette.middleware.cors import CORSMiddleware
from threading import Thread
import torch
from starlette.responses import StreamingResponse
import json
import time

# 设置模型路径
model_path = "/Users/chenwenjie/.cache/modelscope/hub/qwen/Qwen1___5-1___8B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().to('cpu')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.eval()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 流式生成函数，逐步返回 JSON 格式的生成内容
async def generate_stream(query: str):
    inputs = tokenizer.encode(query, return_tensors="pt").to('cpu')
    attention_mask = torch.ones(inputs.shape, device='cpu')

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(inputs=inputs, attention_mask=attention_mask, max_new_tokens=512, streamer=streamer)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        json_data = json.dumps({
            "response": new_text,
            "status": 200
        }) + "\n"
        yield json_data.encode("utf-8")
        time.sleep(0.01)

    thread.join()


@app.post("/generate")
async def generate_text(request: dict):
    query = request.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        return StreamingResponse(generate_stream(query), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    uvicorn.run(app, host="0.0.0.0", port=8005, workers=1)


if __name__ == "__main__":
    main()