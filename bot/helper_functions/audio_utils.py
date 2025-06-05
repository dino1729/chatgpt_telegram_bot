"""
Audio processing, speech recognition, and text-to-speech utilities
"""
import json
import uuid
import requests
import azure.cognitiveservices.speech as speechsdk

class AudioUtils:
    """Utilities for audio processing and speech services"""
    
    def __init__(self, config_manager):
        self.config = config_manager

async def transcribe_audio(audio_file, config_manager):
    """
    Transcribe audio file using Azure Speech Services
    
    Args:
        audio_file: Path to audio file
        config_manager: Configuration manager instance
        
    Returns:
        Tuple of (translated_result, detected_language)
    """
    # Create an instance of a speech config with your subscription key and region
    endpoint_string = f"wss://{config_manager.azurespeechregion}.stt.speech.microsoft.com/speech/universal/v2"
    audio_config = speechsdk.audio.AudioConfig(filename=str(audio_file))
    
    # Set up translation parameters: source language and target languages
    translation_config = speechsdk.translation.SpeechTranslationConfig(
        subscription=config_manager.azurespeechkey,
        endpoint=endpoint_string,
        speech_recognition_language='en-US',
        target_languages=('en', 'hi', 'te')
    )
    
    # Specify the AutoDetectSourceLanguageConfig
    auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
        languages=["en-US", "hi-IN", "te-IN"]
    )
    
    # Creates a translation recognizer using audio file as input
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=translation_config, 
        audio_config=audio_config,
        auto_detect_source_language_config=auto_detect_source_language_config
    )
    
    result = recognizer.recognize_once()
    translated_result = format(result.translations['en'])
    detected_src_lang = format(result.properties[speechsdk.PropertyId.SpeechServiceConnection_AutoDetectSourceLanguageResult])

    return translated_result, detected_src_lang

async def translate_text(text, target_language, config_manager):
    """
    Translate text using Azure Translator
    
    Args:
        text: Text to translate
        target_language: Target language code
        config_manager: Configuration manager instance
        
    Returns:
        Translated text
    """
    # Add your key and endpoint
    key = config_manager.azuretexttranslatorkey
    endpoint = "https://api.cognitive.microsofttranslator.com"
    location = config_manager.azurespeechregion
    
    path = '/translate'
    constructed_url = endpoint + path
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': [target_language]
    }
    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    
    body = [{'text': text}]
    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()
    return response[0]['translations'][0]['text']

async def text_to_speech(text, output_path, language, config_manager):
    """
    Convert text to speech using Azure Speech Services
    
    Args:
        text: Text to convert
        output_path: Output file path
        language: Language code
        config_manager: Configuration manager instance
    """
    speech_config = speechsdk.SpeechConfig(
        subscription=config_manager.azurespeechkey, 
        region=config_manager.azurespeechregion
    )
    
    # Set the voice based on the language
    if language == "te-IN":
        speech_config.speech_synthesis_voice_name = "te-IN-ShrutiNeural"
    elif language == "hi-IN":
        speech_config.speech_synthesis_voice_name = "hi-IN-SwaraNeural"
    else:
        # Use a default voice if the language is not specified or unsupported
        speech_config.speech_synthesis_voice_name = "en-US-NancyNeural"
    
    # Use the default speaker as audio output
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = speech_synthesizer.speak_text_async(text).get()
    
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        # Get the audio data from the result object
        audio_data = result.audio_data  
        # Save the audio data as a WAV file
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_data)

async def local_text_to_speech(text, output_path, model_name, config_manager):
    """
    Convert text to speech using local RVC TTS API
    
    Args:
        text: Text to convert
        output_path: Output file path
        model_name: Voice model name
        config_manager: Configuration manager instance
    """
    url = config_manager.rvctts_api_base
    payload = json.dumps({
        "speaker_name": model_name,
        "input_text": text,
        "emotion": "Angry",
        "speed": 1.5
    })
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        audio_content = response.content
        # Save the audio to a file
        with open(output_path, "wb") as audio_file:
            audio_file.write(audio_content)
    else:
        print("Error:", response.text)
