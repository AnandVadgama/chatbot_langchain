# its include all we want strict schema enforcement, type safety, easy validation, seamless integration
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):

    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)

prompt = PromptTemplate(
    template="generate the name, age, and city of a {place} person \n {format_instructions}",
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    }
)

chain = prompt | model | parser

result = chain.invoke({"place": "gujarati"})

print(result)