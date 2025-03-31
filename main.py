from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from openai import OpenAI
import os
import zipfile
import pandas as pd
import tempfile
from dotenv import load_dotenv 

load_dotenv()


app = FastAPI()

# Load your OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

@app.post("/api/")
async def process_question(
    question: str = Form(...), 
    file: UploadFile = File(None)
):
    # Handle file processing if provided
    if file:
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = os.path.join(tmp_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Extract if ZIP file
            if file.filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_dir)

                # Find CSV and read its content
                for root, _, files in os.walk(tmp_dir):
                    for csv_file in files:
                        if csv_file.endswith('.csv'):
                            csv_path = os.path.join(root, csv_file)
                            df = pd.read_csv(csv_path)
                            if 'answer' in df.columns:
                                answer = df['answer'].iloc[0]
                                return JSONResponse(content={"answer": str(answer)})

    # Query OpenAI API if no CSV is provided
    response = client.chat_completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
        temperature=0.3,
        max_tokens=500
    )

    answer = response.choices[0].message.content.strip()
    
    return JSONResponse(content={"answer": answer})