from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Test</title></head>
    <body>
        <h1>안녕하세요! 이것은 작동합니다!</h1>
        <p>질문을 입력하세요:</p>
        <input type="text" placeholder="질문">
    </body>
    </html>
    """
