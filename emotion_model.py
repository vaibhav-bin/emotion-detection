import torch
from transformers import pipeline


MODEL_ID = (
    "firdhokk/"
    "speech-emotion-recognition-with-openai-whisper-large-v3"
)


DEVICE = (
    0
    if torch.cuda.is_available()
    else -1
)


print("Loading emotion model...")

print(
    "CUDA:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


classifier = pipeline(
    "audio-classification",
    model=MODEL_ID,
    device=DEVICE,
)


print("Emotion model loaded.")


def predict_emotion(audio_path: str):

    results = classifier(
        audio_path
    )

    results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return [
        {
            "emotion":
                item["label"],

            "score":
                float(item["score"]),
        }

        for item in results
    ]