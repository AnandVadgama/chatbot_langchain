from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# prompt template 1
template1 = PromptTemplate(
    template="write a detailed report on a {topic}",
    input_variables=["topic"]
)

# prompt template 2
template2 = PromptTemplate(
    template="write a 5 line summary on the following text.\n {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({ "topic": "black hole"})

print(result)