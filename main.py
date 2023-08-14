import os

from langchain.llms import OpenAI

if __name__ == '__main__':
    openai_api_key = os.environ["OPENAI_API_KEY"]
    print(openai_api_key)
    llm = OpenAI(temperature=0.9)
    prompt = "What would be a good company name for a company than makes colorful socks?"

    result = llm.generate([prompt] * 2)
    for company_name in result.generations:
        print(company_name[0].text)
