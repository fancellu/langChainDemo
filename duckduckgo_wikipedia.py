import pprint

from langchain.agents import load_tools, initialize_agent, AgentType, get_all_tool_names
from langchain.llms import OpenAI

if __name__ == '__main__':
    prompt = "Where are the next summer olympics going to be hosted? What is the population of that country ?"

    pp = pprint.PrettyPrinter(indent=4)

    pp.pprint(get_all_tool_names())

    # We don't want it to get creative!
    llm = OpenAI(temperature=0)

    tools = load_tools(["ddg-search", "wikipedia"])

    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    print(agent.run(prompt))
