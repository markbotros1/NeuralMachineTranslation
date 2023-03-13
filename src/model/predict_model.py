import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import re
import nltk
import torch
from nmt import NMT
from data.vocab import Vocab

def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

def predict(model, input_string):
    nmt_document_preprocessor = lambda x: nltk.word_tokenize(x)
    with torch.no_grad():
        translation = untokenize(model.beam_search(
            nmt_document_preprocessor(input_string),
            beam_size=64,
            max_decoding_time_step=len(input_string)+10
        )[0].value)
    return translation

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = NMT.load("models/mod_ab.ckpt")
    model.to(device)
    model.device = device
    model.decoder.device = device

    input_string = "where did you come from?"
    print(predict(model, input_string))