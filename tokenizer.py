import sys
from speechbrain.tokenizers.SentencePiece import SentencePiece
from hyperpyyaml import load_hyperpyyaml
from prepare_dataset import prepare_mini_librispeech

class Tokenizer:

    yaml_file = open("tokenizer.yaml")
    parsed_yaml_file = load_hyperpyyaml(yaml_file)

    output_folder: str = parsed_yaml_file["output_folder"]
    token_output: int = parsed_yaml_file["token_output"]
    train_annotation: str = parsed_yaml_file["train_annotation"]
    valid_annotation: str = parsed_yaml_file["valid_annotation"]
    annotation_read: str = parsed_yaml_file["annotation_read"]
    token_type: str = parsed_yaml_file["token_type"]
    character_coverage: float = parsed_yaml_file["character_coverage"]

    def __init__(self):
        pass

    def run(self):


        #output_folder = parsed_yaml_file["output_folder"]
        #print(output_folder)
        print("Training the tokenizer...")

        spm = SentencePiece(
            model_dir= self.output_folder,
            vocab_size= self.token_output ,
            annotation_train= self.train_annotation,
            annotation_read= self.annotation_read,
            model_type= self.token_type,
            character_coverage= self.character_coverage,
            annotation_list_to_check= [self.train_annotation, self.valid_annotation],
            annotation_format= "json",
        )

        print(" \n Tokenizer trained successfully!")


# Testing the tekonizer
#import sentencepiece as spm
#sp = spm.SentencePieceProcessor()
#sp.load("dataset/tokenizer/1000_unigram.model")

#print(sp.encode_as_pieces('The city of Tunis'))

