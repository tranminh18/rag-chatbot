from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq()

history = [{"role": "system", "content": "Bạn là trợ lý hữu ích."}]

print("Chat với AI (gõ 'quit' để thoát)")
while True:
    user_input = input("Bạn: ")
    if user_input == "quit":
        break

    history.append({"role": "user", "content": user_input})

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=history
    )

    reply = res.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    print(f"AI: {reply}")