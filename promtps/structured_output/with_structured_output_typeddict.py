# only for type hinting and basic structure 

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict, Annotated, Optional, Literal
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

class Review(TypedDict):
    """
    key ropics disccussed in the review
    summary of the review
    sentiment of the review
    pros and cons of the review
    name of the reviewer
    """
    key_themes: Annotated[list[str], "key topics discussed in the review"]
    summary: Annotated[str, "summary of the review"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "sentiment of the review"]
    pros: Annotated[Optional[list[str]], "pros of the review"]
    cons: Annotated[Optional[list[str]], "cons of the review"]
    name: Annotated[Optional[str], "reviewed by name"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

                                                                 
reviever name: Anand Vadgama
""")

print(result['name'])