# LangChain demo project

## To install requirements

```pip install -r requirements.txt```

## To run

Make sure you have environmental variable setup for OpenAI and HuggingFace API keys

| Keyname                  |
|--------------------------|
| HUGGINGFACEHUB_API_TOKEN |
| OPENAI_API_KEY           |

### [main.py](main.py)

A simple OpenAI prompt example. e.g.
```
> What would be a good company name for a company than makes colorful socks?
Rainbow Socksy
BrightToes Socks
```

### [hf.pay](hf.py)

A hugging face prompt example using the flan-t5-base model
```
> translate English to German: How old are you?
Wie alt sind Sie?
```

### [template_and_chain.py](template_and_chain.py)
Example of prompt templating and prompt chaining
```
> You are a naming consultant for new companies. What is a good name for a company that makes {product}?
> colorful socks
Rainbow Socks Co.
> Digital pianos
DigitalPianoKeys.
```
```
> You are a naming consultant for new companies. What is a good name for a {company} that makes {product}?
> {"company": "AI Startup", "product": "large language models"}
BigTalk Technologies.
```

### [simple_sequential_chain.py](simple_sequential_chain.py)
Another way to do simple chaining
```
> What is a good name for a company that makes {product}?
> Write a catch phrase for the following company: {company_name}
> Gaming mice
"Level up with GamersRigged!"
```

### [action.py](action.py)
How to supply extra tools/plugins/agents to llm to perform certain functionality
```
> When was the 3rd president of the USA born? What is that year cubed?
Thomas Jefferson was born in 1743 and the year cubed is 5295319407
```

### [duckduckgo_wikipedia.py](duckduckgo_wikipedia.py)
Another example of action agents
```
> Where are the next summer olympics going to be hosted? What is the population of that country ?
The 2024 Olympics will be held in Paris, France, with a population of 68,035,000 people
```
 

### [plan.py](plan.py)
Complex PlanAndExecute agent example
```
> Where are the next summer olympics going to be hosted? What is the population of that country, squared?
The square of the population of Paris, France is 4,840,000,000,000
```

### [chat.py](chat.py)
A simple chat repl
```
>  Why do dogs have black lips?
Dogs have black lips to help protect their skin from the sun's UV rays. The black pigment in their lips helps to absorb the sun's rays and prevent sunburn.
> And how many legs do they generally have?
Dogs generally have four legs

```

### [documentQA.py](documentQA.py)
Loads up a text document (State of the Union 2020) into chromadb, and lets you do QA/chat with LLM about that document
```
> Summarize what Trump says
President Trump states that since he took office, the U.S. economy has improved, with 7 million new jobs created, and the country is thriving and highly respected again. He has taken action to combat human trafficking and secure the southern border, ending the practice of catch and release.
> Make a list of all his talking points
- Reviving the US economy
- Slashing job-killing regulations
- Enacting historic and record-setting tax cuts
- Fighting for fair and reciprocal trade agreements
- Creating 7 million new jobs
- Putting America first
- Protecting patients with pre-existing conditions
- Protecting Medicare and Social Security
- Requiring price transparency
- Replacing the disastrous NAFTA trade deal
```

### [flask_documentQA.py](documentQA.py)
Same as above, but with a web UI