import uvicorn
from starlette.websockets import WebSocket, WebSocketDisconnect
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from threading import Thread
import torch
import time

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            json_request = await websocket.receive_json()
            query = json_request.get("query")
            history = json_request.get("history", [])

            if not query:
                await websocket.send_json({'error': 'Query cannot be empty', 'status': 400})
                continue

            try:
                inputs = tokenizer.encode(query, return_tensors="pt").to('cpu')
                attention_mask = torch.ones(inputs.shape, device='cpu')

                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_kwargs = dict(inputs=inputs, attention_mask=attention_mask, max_new_tokens=512, streamer=streamer)
                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                for new_text in streamer:
                    json_data = {
                        "response": new_text,
                        "status": 200
                    }
                    await websocket.send_json(json_data)
                    time.sleep(0.1)

                thread.join()

            except Exception as e:
                await websocket.send_json({'error': str(e), 'status': 500})

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    finally:
        if websocket.client_state == 1:
            await websocket.close()

def main():
    uvicorn.run(app, host="0.0.0.0", port=8005, workers=1)

if __name__ == "__main__":
    main()