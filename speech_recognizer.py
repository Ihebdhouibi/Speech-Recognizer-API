from fastapi import FastAPI
from prepare_dataset import prepare_mini_librispeech, skip
from tokenizer import Tokenizer

app = FastAPI()
@app.get("/")
async def root():
    try:
        prepare_mini_librispeech("dataset/",
                                 "dataset/specification/train.json",
                                 "dataset/specification/valid.json",
                                 "dataset/specification/test.json")
    except Exception as e:
        print("There has been an error in preparing dataset " + str(e))

    if not skip("dataset/tokenizer/1000_unigram.model"):
        try:
            Tk = Tokenizer()
            Tk.run()
        except Exception as e:
            print("There had been an error training the tokenizer "+ str(e))

    else:
        print("Tokenizer already trained")

    return {"message": "API Created successfully !!!"}

