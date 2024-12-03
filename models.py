import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pickle

def getAvailableLanguages():
    """Get available languages for the Silero models."""
    available_languages = torch.hub.load('snakers4/silero-models', 'available_languages')
    return available_languages

def getASRModel(language: str) -> nn.Module:
    """Get Speech-to-Text (ASR) model based on language."""
    
    if language == 'de':
        # German Speech-to-Text model (Silero)
        model, decoder, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_stt',
            language='de',
            device=torch.device('cpu')
        )
    elif language == 'en':
        # English Speech-to-Text model (Silero)
        model, decoder, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_stt',
            language='en',
            device=torch.device('cpu')
        )
    # elif language == 'fr':
    #     # Use Wav2Vec2 model for French ASR
    #     model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
    #     processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
    #     # Save model and processor for reuse
    #     with open('wav2vec2_model_fr.pickle', 'wb') as handle:
    #         pickle.dump(model, handle)
    #     with open('wav2vec2_processor_fr.pickle', 'wb') as handle:
    #         pickle.dump(processor, handle)
    #     # Return the model and processor
    #     return model, processor
    elif language == 'fr':
        # Charger le modèle français
        model_name = "bofenghuang/asr-wav2vec2-ctc-french"
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        return model, processor
    
    else:
        raise ValueError(f'Language "{language}" not supported for Speech-to-Text.')

    return model, decoder

def getTTSModel(language: str) -> nn.Module:
    """Get Text-to-Speech (TTS) model based on language."""
    if language == 'de':
        # German Text-to-Speech model (Silero)
        speaker = 'thorsten_v2'  # 16 kHz
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker=speaker
        )
    elif language == 'en':
        # English Text-to-Speech model (Silero)
        speaker = 'lj_16khz'  # 16 kHz
        model = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker=speaker
        )
    elif language == 'fr':
        # French Text-to-Speech model (choose from fr_0 to fr_5 or 'random')
        speaker = 'fr_0'  # Choose a speaker like fr_0, fr_1, etc.
        model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=language,
            speaker=speaker
        )
    else:
        raise ValueError(f'Language "{language}" not supported for Text-to-Speech.')

    return model

def getTranslationModel(language: str) -> nn.Module:
    """Get translation model for a given language."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    if language == 'de':
        # German translation model
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
        # Cache models to avoid Hugging Face processing
        with open('translation_model_de.pickle', 'wb') as handle:
            pickle.dump(model, handle)
        with open('translation_tokenizer_de.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle)

    elif language == 'fr':
        # French translation model (French -> English)
        model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
        # Cache models to avoid Hugging Face processing
        with open('translation_model_fr.pickle', 'wb') as handle:
            pickle.dump(model, handle)
        with open('translation_tokenizer_fr.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle)

    else:
        raise ValueError(f'Language "{language}" not supported for translation.')

    return model, tokenizer
