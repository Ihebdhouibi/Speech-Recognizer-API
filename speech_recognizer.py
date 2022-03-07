from fastapi import FastAPI
from prepare_dataset import prepare_mini_librispeech

app = FastAPI()
@app.get("/")
async def root():
    try:
        prepare_mini_librispeech("dataset/",
                                 "dataset/json-train.txt",
                                 "dataset/json-valid.txt",
                                 "dataset/json-test.txt")
    except Exception as e:
        print("There has been an error in preparing dataset " + str(e))

    return {"message": "API Created succfvessfully !!!"}

