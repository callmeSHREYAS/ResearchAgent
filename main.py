from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_agent
load_dotenv()

class ResearchResponse(BaseModel):
    topic:str
    summary:str
    sources:list[str]
    tools_used:list[str]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)



format_instructions=parser.get_format_instructions()
agent = create_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[],
    system_prompt=(
        """You are a research assistant that helps generate a research paper. 
        Answer the user query and use tools when necessary."""
    ),
    response_format=ResearchResponse
)


result = agent.invoke({"messages": [{"role": "user", "content": "how to make girlfriend"}]})
structured = result["structured_response"]   # a ResearchResponse object

# print(structured.topic)
# print(structured.summary)
print(structured.sources[0])
# print(structured.tools_used)

# # or as dict / JSON
# print(structured.model_dump())
# print(structured.model_dump_json(indent=2))

print(result)