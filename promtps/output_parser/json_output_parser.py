# in this the problem is we don't have wanted structure output

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

prompt = PromptTemplate(
    template="give me 5 facts about {topic}. \n {formate_instructions}",
    input_variables=["topic"],
    partial_variables={
        "formate_instructions": parser.get_format_instructions()
    }
)

chain = prompt | model | parser

result = chain.invoke({ "topic": "black hole"})

print(result)
print(type(result))