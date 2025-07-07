# we have wanted structure output but its not including data validation

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-0.6B",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact-1", description="fact 1 about black hole"),
    ResponseSchema(name="fact-2", description="fact 2 about black hole"),
    ResponseSchema(name="fact-3", description="fact 3 about black hole")
]

parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
    template="give me 3 facts about {topic} , {format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    }
)

chain = prompt | model | parser

result = chain.invoke({"topic": "black hole"})

print(result)