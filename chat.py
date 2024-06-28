import chainlit as cl
from typing import List
from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an AI assistant that gives helpful answers in a brief and concise manner."
    prompt = f"\n### System:\n{system}\n\n### User:\n"

    # if history is not None:
    if len(history) > 0:
        prompt += f"This is the conversation history:{''.join(history)}. \nNow answer the question: "

    prompt += f"{instruction}\n\n### Response:"
    # print(prompt)
    return prompt


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    response = ""

    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word

    await msg.update()
    message_history.append(response)


@cl.on_chat_start
def on_chat_start():
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )

    cl.user_session.set("message_history", [])


"""
history = []

question = "Which is the capital city of India?"

answer = ""
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "And which is of the United Kingdom?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
"""
