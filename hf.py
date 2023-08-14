import os

from langchain.llms import HuggingFaceHub

if __name__ == '__main__':
    huggingfacehub_api_token = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    print(huggingfacehub_api_token)

    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0, "max_length": 64})
    # https://huggingface.co/google/flan-t5-base

    prompt = "What are good fitness tips?"

    print(llm(prompt))
