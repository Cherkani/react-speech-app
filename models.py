import torch
import torch.nn as nn

import pickle


import pickle


def getASRModel(language: str) -> nn.Module:

    if language == 'de':

        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='de',
                                               device=torch.device('cpu'))

    elif language == 'en':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='en',
                                               device=torch.device('cpu'))
    elif language == 'fr':
        model, decoder, utils = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                               model='silero_stt',
                                               language='fr',
                                               device=torch.device('cpu'))

    return (model, decoder)

# import torch
# from torch import nn
# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# def getASRModel(language: str) -> nn.Module:
#     if language == 'de':
#         # Load the Silero model for German
#         model, decoder, utils = torch.hub.load(
#             repo_or_dir='snakers4/silero-models',
#             model='silero_stt',
#             language='de',
#             device=torch.device('cpu')
#         )

#     elif language == 'en':
#         # Load the Silero model for English
#         model, decoder, utils = torch.hub.load(
#             repo_or_dir='snakers4/silero-models',
#             model='silero_stt',
#             language='en',
#             device=torch.device('cpu')
#         )
        
#     elif language == 'fr':
#         # Load Wav2Vec 2.0 model for French from Hugging Face
#         model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53-french')
#         processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-xlsr-53-french')
        
#         # The processor can serve as the decoder to convert model output to text
#         decoder = processor

#     return model, decoder



def getTTSModel(language: str) -> nn.Module:

    if language == 'de':

        speaker = 'thorsten_v2'  # 16 kHz
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                  model='silero_tts',
                                  language=language,
                                  speaker=speaker)

    elif language == 'en':
        speaker = 'lj_16khz'  # 16 kHz
        model = torch.hub.load(repo_or_dir='snakers4/silero-models',
                               model='silero_tts',
                               language=language,
                               speaker=speaker)
    else:
        raise ValueError('Language not implemented')

    return model


def getTranslationModel(language: str) -> nn.Module:
    from transformers import AutoTokenizer
    from transformers import AutoModelForSeq2SeqLM
    if language == 'de':
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en")
        tokenizer = AutoTokenizer.from_pretrained(
            "Helsinki-NLP/opus-mt-de-en")
        # Cache models to avoid Hugging face processing
        with open('translation_model_de.pickle', 'wb') as handle:
            pickle.dump(model, handle)
        with open('translation_tokenizer_de.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle)
    else:
        raise ValueError('Language not implemented')

    return model, tokenizer
