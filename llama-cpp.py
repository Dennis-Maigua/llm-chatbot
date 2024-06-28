from llama_cpp import Llama
import os


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# $ huggingface-cli download [gguf_id] [gguf_file] --local-dir . --local-dir-use-symlinks False
MODEL_PATH = "./models/medical-llama3-8b.Q5_K_M.gguf"


B_INST, E_INST = "<s>[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
    You are an AI Medical Personal Assistant equipped with extensive medical knowledge.
    You aim to provide responses to your inquiries in an accurate and concise manner.
    However, your responses should not be considered a replacement for professional medical advice.
    If you do not know the answer to a question, please refrain from providing any inaccurate information.
"""
SYSTEM_PROMPT = B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS


def create_prompt(user_query):
    instruction = f"User asks: {user_query}\n"
    prompt = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt.strip()


user_query = "Hello, I'm a 35-year-old male experiencing symptoms like fatigue, increased sensitivity to cold, and dry, itchy skin."
prompt = create_prompt(user_query)
print(prompt)

llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1)
result = llm(
    prompt=prompt,
    max_tokens=100,
    echo=False
)
print(result['choices'][0]['text'])
