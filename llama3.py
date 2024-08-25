from ollama import Client



client = Client(host='http://localhost:11434')

def text_generation(request: str) -> str:
    print(f"Request to Ollama: {request}")
    try:
        response = client.chat(model='llama3.1', messages=[
          {
            'role': 'user',
            'content': request,
          },
        ])
        return response["message"]["content"]
    except Exception as e:
        raise e
