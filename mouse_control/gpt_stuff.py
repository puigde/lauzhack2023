# of course we will use LLMs its a 2023 hackathon
from openai import OpenAI

with open("../OPENAI_API_KEY.txt", "r") as f:
    OPENAI_API_KEY = f.read().strip()
print(OPENAI_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


def gpt_explain(text: str) -> str:
    """Explain a text using GPT-3.5 turbo"""
    prompt = f"""
    Explain the following text:
    "{text}"
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{prompt}"},
        ],
    )
    return response.choices[0].message.content
