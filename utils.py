import regex as re
import torch 
import pickle
from tokenizers import BertWordPieceTokenizer



def word_piece_decoder(text):
    # Reverse of preProcessing specific to word_piece
    text = re.sub('a',r'\u094D',text)
    text = re.sub('y',r'\u0941',text)
    text = re.sub('e',r'\u0942',text)
    text = re.sub('i',r'\u0901',text)
    text = re.sub('c',r'\u0902',text)
    text = re.sub('r',r'\u0943',text)
    text = re.sub('l',r'\u0947',text)
    text = re.sub('o',r'\u094b',text)
    text = re.sub('p',r'\u094c',text)
    text = re.sub('k',r'\u0948',text)
    
    return text


def word_piece_encoder(text):
    #replace devnagari tokens that doesn't work well for the bert-wordpiece tokenizer    
    text = re.sub(r'\u094D','a',text)
    text = re.sub(r'\u0941','y',text)
    text = re.sub(r'\u0942','e',text)
    text = re.sub(r'\u0901','i',text)
    text = re.sub(r'\u0902','c',text)
    text = re.sub(r'\u0943','r',text)
    text = re.sub(r'\u0947','l',text)
    text = re.sub(r'\u094b','o',text)
    text = re.sub(r'\u094c','p',text)
    text = re.sub(r'\u0948','k',text)
    
    return text

def morpheme_encoder(text):
    
    # Trick$ to make the morpheme$ work together
    tr = re.sub(r'[ ]+', r' ', text)
    tr = re.sub(r'- -', r'*', tr)
    tr = re.sub(r'-[ ]+', r'*', tr)
    tr = re.sub(r'[ ]+-', r'*',tr)
    tr = re.sub(r'\*', r' * ',tr)
    tr = re.sub(r'-', r' ',tr)
    
    return tr
    
    


def preProcessText(text, tokenizer_type = 'default'):
    
    
    if tokenizer_type == 'morpheme':
#         text = re.sub(r'\s*[\u0966-\u0976]+\s*', '\u0020<num>\u0020', text)
        text = morpheme_encoder(text)
        
        return text
    
    
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
    
    if tokenizer_type == 'default':
        # add space in between the devanagari numbers and replace number by <num> token
        text = re.sub(r'\s*[\u0966-\u0976]+\s*', '\u0020<num>\u0020', text)
    
    
    elif tokenizer_type == 'word_piece':
        #replace devnagari tokens that doesn't work well for the bert-wordpiece tokenizer  
        text = word_piece_encoder(text)
        
    elif tokenizer_type == 'morpheme':
#         text = re.sub(r'\s*[\u0966-\u0976]+\s*', '\u0020<num>\u0020', text)
        text = morpheme_encoder(text)
        
    elif tokenizer_type == 'sentence_piece':
        pass
    
    elif tokenizer_type == 'morph_bpe':
        text = morpheme_encoder(text)
        
    
        
    return text


def getTokenizer(tokenizer_type = 'default'):
    tokenizer_dir = "tokenizers"
    
    
    if tokenizer_type == 'default':
        tokenizer_path = tokenizer_dir + "/tokenizer.pth"
        vocab_path = tokenizer_dir + "/vocab.pkl"
        loaded_tokenizer = torch.load(tokenizer_path)
        with open(vocab_path, 'rb') as file:
            loaded_vocab = pickle.load(file)
            
    elif tokenizer_type == 'word_piece':
        tokenizer_path = tokenizer_dir + "/tokenizer_wp.pickle"
    
    elif tokenizer_type == 'sentence_piece':
        tokenizer_path = tokenizer_dir + "/tokenizer_sp.pickle"
 
    elif tokenizer_type == 'morph_bpe':
        tokenizer_path = tokenizer_dir + "/tokenizer_mp_bpe.pickle"
        
    elif tokenizer_type == 'morpheme':
        tokenizer_path = tokenizer_dir + "/tokenizer.pth"
        vocab_path = tokenizer_dir + "/transformer_vocab_morpheme.pickle"
        loaded_tokenizer = torch.load(tokenizer_path)
        with open(vocab_path,'rb') as f:
            loaded_vocab = pickle.load(f)
    
    if tokenizer_type != 'default' and tokenizer_type!='morpheme':
        with open(tokenizer_path, 'rb') as file:
            loaded_tokenizer = pickle.load(file)
            
        loaded_vocab = loaded_tokenizer.get_vocab()

    return loaded_tokenizer, loaded_vocab

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')



def split_list(l):
    splitted_list = []
    z = 0
    for i,idx in enumerate(l):
        if idx == 220:
            splitted_list.append(l[z:i])
            z = i+1
    if z <= len(l)-1:
        splitted_list.append(l[z:])
    return splitted_list