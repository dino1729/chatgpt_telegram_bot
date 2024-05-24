from openai import OpenAI

client = OpenAI(
    base_url = 'http://10.0.0.164:11434/api/v1',
    api_key='ollama', # required, but unused
)

response = client.chat.completions.create(
  model="mistral",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The LA Dodgers won in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response.choices[0].message.content)

chat_completion = client.chat.completions.create(
    messages=[
        {
            'role': 'user',
            'content': 'Say this is a test',
        }
    ],
    model='mistral',
)
print(chat_completion.choices[0].message.content)

response2 = client.chat.completions.create(
  model="mistral", 
  messages=[
    {
        "role": "system", 
        "content": "You are a professional content creation assistant."},
    {
        "role": "user", 
        "content": "In 1 sentence, what is the best kind of content to create in 2024?"
    },
  ]
)
print(response2.choices[0].message.content)
