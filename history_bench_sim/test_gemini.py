import google.generativeai as genai

model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    system_instruction="You are a helpful assistant"
)

chat = model.start_chat(history=[])

response = chat.send_message(["How are you?"])
print(response.text)
