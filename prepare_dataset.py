import os
import json
import shutil
import logging
from speechbrain.utils.data_utils import get_all_files, download_file
from speechbrain.dataio.dataio import read_audio

logger = logging.getLogger(__name__)
MINILIBRI_TRAIN_URL = "http://www.openslr.org/resources/31/train-clean-5.tar.gz"
MINILIBRI_VALID_URL = "http://www.openslr.org/resources/31/dev-clean-2.tar.gz"
MINILIBRI_TEST_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
SAMPLERATE = 16000


# skip data preparation if already done
def skip(*filenames: tuple):
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


# check dataset downloaded or not
def check_folders(*folders: tuple):
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


# download dataset
def download_mini_librispeech(destination: str):
    train_archive = os.path.join(destination, "train-clean-5.tar.gz")
    valid_archive = os.path.join(destination, "dev-clean-2.tar.gz")
    test_archive = os.path.join(destination, "test-clean.tar.gz")

    download_file(MINILIBRI_TRAIN_URL, train_archive)
    download_file(MINILIBRI_VALID_URL, valid_archive)
    download_file(MINILIBRI_TEST_URL, test_archive)

    shutil.unpack_archive(train_archive, destination)
    shutil.unpack_archive(valid_archive, destination)
    shutil.unpack_archive(test_archive, destination)


# returns transcription of each sentence in the dataset
def get_transcription(trans_list: list[str]):
    trans_dict = {}
    for trans_file in trans_list:
        with open(trans_file) as f:
            for line in f:
                uttid = line.split(" ")[0]
                text = line.rstrip().split(" ")[1:]
                text = " ".join(text)
                trans_dict[uttid] = text

    logger.info("Transcription files read!")
    return trans_dict


# create json files
def create_json(wav_list: list[str], trans_dict, json_file: str):
    json_dict = {}
    for wav_file in wav_list:
        # retreiving the duration through reading the signal, duration in seconds
        signal = read_audio(wav_file)
        duration = signal.shape[0] / SAMPLERATE

        # get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        relative_path = os.path.join("{data_root}", *path_parts[-5:])

        # create entry for this utterance
        json_dict[uttid] = {
            "wav": relative_path,
            "length": duration,
            "words": trans_dict[uttid],
        }

    # writting the dictionary to json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created")


def prepare_mini_librispeech(
        data_folder: str, save_json_train: str, save_json_valid: str, save_json_test: str
):
    # Checking if data prepation is already done
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Data preparation is already done. ")
        return

    # Data preparation
    # if the dataset doesn't exist, DOWNLOAD
    train_folder = os.path.join(data_folder, "Librispeech", "train-clean-5")
    valid_folder = os.path.join(data_folder, "Librispeech", "dev-clean-2")
    test_folder = os.path.join(data_folder, "Librispeech", "test-clean")
    if not check_folders(train_folder, valid_folder, test_folder):
        download_mini_librispeech(data_folder)

    # list files and create manifest from lists
    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, {save_json_test}"
    )
    extension = [".flac"]

    # list all flac audio files
    wav_list_train = get_all_files(train_folder, match_and=extension)
    wav_list_valid = get_all_files(valid_folder, match_and=extension)
    wav_list_test = get_all_files(test_folder, match_and=extension)

    # list transcription files
    extension = [".trans.txt"]
    trans_list = get_all_files(data_folder, match_and=extension)
    trans_dict = get_transcription(trans_list)

    # Create json file
    create_json(wav_list_train, trans_dict, save_json_train)
    create_json(wav_list_valid, trans_dict, save_json_valid)
    create_json(wav_list_test, trans_dict, save_json_test)

