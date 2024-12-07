from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def getASRModel(language: str) -> nn.Module:
    if language == 'fr':
        # Load Wav2Vec 2.0 model for French from Hugging Face
        model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-xlsr-53-french')
        processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-xlsr-53-french')
        
        # The processor can serve as the decoder to convert model output to text
        decoder = processor

    return model, decoder


# Load the model and decoder (processor)
language = 'fr'
model, decoder = getASRModel(language)

# Load an example audio file (e.g., wav file)
audio_input, _ = librosa.load('path_to_audio.wav', sr=16000)

# Use the processor to prepare the audio input
inputs = decoder(audio_input, return_tensors="pt", padding=True)

# Get model predictions
with torch.no_grad():
    logits = model(input_values=inputs.input_values).logits

# Decode the logits to text
transcription = decoder.batch_decode(logits.argmax(dim=-1))

print("Transcription:", transcription)
