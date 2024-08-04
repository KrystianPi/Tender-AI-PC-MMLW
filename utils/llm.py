import os
from openai import OpenAI
from dotenv import load_dotenv

class LLM():
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key is None:
            raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable in the .env file.")
        else:
            self.client = OpenAI(api_key=self.api_key)

    def complete(self, prompt):        
        completion = self.client.chat.completions.create(
                                                model="gpt-4o",
                                                temperature=0.0,
                                                max_tokens=4096,
                                                messages=[{"role": "user", "content": prompt}]
                                                )
        return completion.choices[0].message.content