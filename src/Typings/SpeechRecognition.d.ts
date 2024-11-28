
interface RecognitionData {
  [index: number]: any;
  confidence: number;
  transcript: string;
}

interface SpeechRecognitionEvent {
  results: Array<RecognitionData>;
}
