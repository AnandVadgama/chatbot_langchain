from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate(
    ('system', 'You are a helpful AI assistant.'),
    ('human', '{input}'),
)
chat_history = []
# load chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
prompt = chat_template.invoke({'chat_history':chat_history, 'query':'Where is my refund'})

print (prompt)