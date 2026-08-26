import torch
from transformers import pipeline


MODEL_ID = (
    "firdhokk/"
    "speech-emotion-recognition-with-openai-whisper-large-v3"
)

device = 0 if torch.cuda.is_available() else -1

classifier = pipeline(
    "audio-classification",
    model=MODEL_ID,
    device=device,
)

result = classifier("Young_Female_South.wav")

for item in result:
    print(
        item["label"],
        round(item["score"], 4)
    )