from fastapi import FastAPI

# local imports
from prepare_dataset import prepare_mini_librispeech, skip
from tokenizer import Tokenizer
from language_model import LM
from ASR import ASR
app = FastAPI()


@app.get("/{audio_path}")
async def root(audio_path: str):

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
            print("There had been an error training the tokenizer " + str(e))

    else:
        print("Tokenizer already trained.")

    if not skip("dataset/RNNLM/save/CKPT+2022-03-08+22-05-40+00/model.ckpt"):
        try:
            language_model = LM()
            language_model.run()
        except Exception as e:
            print("There has been an error while training the Language model " + str(e))
    else:
        print("Language model already trained.")

    try:
        asr = ASR()
        asr.run()
    except Exception as e:
        print("There had been an error training ASR " + str(e))

    # Loading audio file to test

    return {"message": "API Created successfully !!!"+str(audio_path)}
