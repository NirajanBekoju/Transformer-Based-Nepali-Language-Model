import regex as re
import torch 
import pickle

def preProcessText(text):
    # put space in beteen the | -> devanagari danda to make it a separate word.
    text = re.sub(r'\s*[\u0964]\s*', r'\u0020\u0964\u0020', text)
    # put space around the question mark ?  to make it a separate word
    text = re.sub(r'\s*[\u003f]\s*', r'\u0020\u003f\u0020', text)
    # put space in between comma(,)
    text = re.sub(r'\s*[\u002c]\s*', r'\u0020\u002c\u0020', text)
    # remove space around the new line character
    text = re.sub(r'\s*\n\s*','\n', text)
    # replace any non-devangari string with a blank
    text = re.sub(r'[^\u0900-\u097F,?\s+]','', text) 
    # add space in between the devanagari numbers and replace number by <num> token
    text = re.sub(r'\s*[\u0966-\u0976]+\s*', '\u0020<num>\u0020', text)
    return text

def getTokenizer():
    tokenizer_dir = "tokenizer"
    tokenizer_path = tokenizer_dir + "/tokenizer.pth"
    vocab_path = tokenizer_dir + "/vocab.pkl"
    loaded_tokenizer = torch.load(tokenizer_path)
    with open(vocab_path, 'rb') as file:
        loaded_vocab = pickle.load(file)

    return loaded_tokenizer, loaded_vocab