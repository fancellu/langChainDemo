# LangChain demo project

## To install requirements

```pip install -r requirements.txt```

## To run

Make sure you have environmental variable setup for OpenAI and HuggingFace API keys

| Keyname                  |
|--------------------------|
| HUGGINGFACEHUB_API_TOKEN |
| OPENAI_API_KEY           |

### main.py

A simple OpenAI prompt example

### hf.pay

A hugging face prompt example using the flan-t5-base model

### template_and_chain.py

Example of prompt templating and prompt chaining

### simple_sequential_chain.py

Another way to do simply chaining

### action.py

How to supply extra tools/plugins/agents to llm to perform certain functionality

### plan.py

Complex PlanAndExecute agent example

### chat.py

A simple chat repl


### documentQA.py

Loads up a text document into chromadb, and lets you do QA/chat with LLM about that document
