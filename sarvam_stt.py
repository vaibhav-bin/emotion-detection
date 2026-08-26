import os

from sarvamai import SarvamAI


SARVAM_API_KEY = os.environ.get(
    "SARVAM_API_KEY"
)

if not SARVAM_API_KEY:
    raise RuntimeError(
        "SARVAM_API_KEY environment variable is not set."
    )


client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)


def transcribe_audio(
    audio_path: str,
    language_code: str = "unknown",
    mode: str = "transcribe",
):

    with open(audio_path, "rb") as audio_file:

        response = client.speech_to_text.transcribe(
            file=audio_file,
            model="saaras:v3",
            language_code=language_code,
            mode=mode,
        )

    return {
        "transcript": response.transcript,
        "language_code": response.language_code,
        "request_id": response.request_id,
    }