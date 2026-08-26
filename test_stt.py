from sarvam_stt import transcribe_audio

result = result = transcribe_audio(
    "0.wav",
    language_code="unknown",
    mode="codemix"
)

print(result["transcript"])

# print(result)