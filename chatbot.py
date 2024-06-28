from typing import List
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "mradermacher/EEVE-korean-medical-chat-10.8b-GGUF", model_file="EEVE-korean-medical-chat-10.8b.Q4_K_M.gguf"
)


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = """\
        You are an AI Medical Personal Assistant equipped with extensive medical knowledge.
        You aim to provide responses to your inquiries in an accurate and concise manner.
        However, your responses should not be considered a replacement for professional medical advice.
    """
    prompt = f"\n### System:\n{system}\n\n### User:\n"

    if history is not None:
        prompt += f"Previous conversation history:{''.join(history)}. \n\nUser asks: "

    prompt += f"{instruction}\n\n### Response:"
    print(prompt)
    return prompt


history = []

question = "Hello, I am a 27-year old experiencing symptoms like fever, cough, vomiting, and diarrhoea."

answer = ""
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "Which medication should I take?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
