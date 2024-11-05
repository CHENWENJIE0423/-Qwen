import uvicorn
from starlette.websockets import WebSocket, WebSocketDisconnect
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

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
                outputs = model.generate(inputs)
                # 这里直接将输出解码为中文字符串
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 将response作为JSON返回
                await websocket.send_json({
                    'response': response,
                    'history': history,
                    'status': 200,
                })
            except Exception as e:
                await websocket.send_json({'error': str(e), 'status': 500})

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    finally:
        await websocket.close()

def main():
    uvicorn.run(app, host="0.0.0.0", port=8009, workers=1)

if __name__ == "__main__":
    main()