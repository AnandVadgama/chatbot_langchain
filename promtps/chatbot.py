from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation"
)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant."),
]
model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting the chatbot. Goodbye!")
        break

    response = model.invoke(user_input)
    chat_history.append(AIMessage(content=response.content))
    print(f"Chatbot: {response.content}")