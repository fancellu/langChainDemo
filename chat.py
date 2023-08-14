from langchain import OpenAI, ConversationChain
from langchain.memory import ChatMessageHistory

if __name__ == '__main__':
    llm = OpenAI(temperature=0)

    # ChatMessageHistory can take previous chat history
    history = ChatMessageHistory()

    conversation = ConversationChain(llm=llm)

    while True:
        my_input = input("Human: ")
        if my_input == "exit":
            break
        if my_input == "dumpchat":
            for m in history.messages:
                print(m)
            continue
        history.add_user_message(my_input)
        ai_reply = conversation.predict(input=my_input)
        history.add_ai_message(ai_reply)
        print(f"AI: {ai_reply}")
