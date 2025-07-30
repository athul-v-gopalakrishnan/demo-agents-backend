from dotenv import load_dotenv
from typing import Sequence
from pydantic import BaseModel,Field
from typing_extensions import Annotated, TypedDict, List

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage,BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from db_config import DB_CONNECTION_STRING
from rag_chatbot.embed_data import rag_vector_store

load_dotenv()

class State(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]

class Summary(BaseModel):
  advantages : str =Field(description="The benefits for the company if the company decides to get into \
                          this agreement with the other parties.")
  disadvantages : str = Field(description="The potential risks and disadvantages for the company \
                              if the company decides to get into this agreement with the other parties.")
  score : int = Field(description="The score from 1 to 10 indicating how good is this deal for the party,\
                       10 meaning fully advantageous with no potential risk and 0 being worst deal to get into.")
  
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
memory = MemorySaver()

search = TavilySearch(max_results = 3)
search.name = "search_tool"
search_node = ToolNode(tools=[search])

summariser_llm = init_chat_model("openai:gpt-4.1", temperature=0.3)
summariser_llm_with_structured_output = summariser_llm.with_structured_output(schema=Summary)
chatbot = init_chat_model("openai:gpt-4.1", temperature=0)

@tool(response_format="content_and_artifact", description="Retrieve data about the agreement from the vector store to answer the query of the user.")
def retrieve(query:str):
    # retriever = PGVector(
    #         embeddings=embedding_model,
    #         collection_name="embeddings",
    #         connection=DB_CONNECTION_STRING
    #     )
    retriever = rag_vector_store
    retrieved_docs = retriever.similarity_search(query = query, k = 5)
    serialised = "\n\n".join(
        (f"Source : {doc.metadata}\nContent:{doc.page_content}")
        for doc in retrieved_docs
    )
    return serialised, retrieved_docs
    
retrieve_node = ToolNode(tools = [retrieve])
chatbot_with_tools = chatbot.bind_tools(tools = [retrieve,search], tool_choice="auto")

summariser_prompt_template = ChatPromptTemplate.from_template(
    "You are the chief strategist of a company. You will be provided with the content from all the agreements\
        to be signed by the company for analysis. You should analyse the document and detect possible risks \
            and the loopholes in the agreement for the advantage of your company in the deal.\n \
                Answer using the schema provided in the class 'Summary'.\n \
                fill each of the entries in the answer schema using multiple bullet points if there are \
                multiple answers.\n \
                Be professional short and on the point, no blabbering. \
                If there are no either advantages or disadvantages return 'Nothing to show' in the corresponding fields.\
                This is the content of the document : \
                {document} \
                You work for {party}."
)

# chatbot_prompt_template = ChatPromptTemplate.from_messages(messages=[("system","You are a legal consultant for {party}.\
#                                                             When asked about the legal sides of the agreements they are provided with \
#                                                             you should use the 'retrive' tool to get the details from the database and answer \
#                                                             the question. When you feel like the knowledge is limited or your knowledge about \
#                                                             something maybe outdated use the 'search_tool' to look up for the latest updates ... \
#                                                             for example about some new ammendments to the laws ..."),
#                                                             MessagesPlaceholder("messages")])

def analyze_document(content,party):
    prompt = summariser_prompt_template.invoke({
        "document":content,
        "party":party
    })
    response = summariser_llm_with_structured_output.invoke(prompt)
    return {
        "advantages":response.advantages,
        "disadvantages":response.disadvantages,
        "score":response.score
    }

def invoke_llm(state:State):
    response = chatbot_with_tools.invoke(state["messages"])
    return {
        "messages":[response]
    }

def conditional_router(state:State):
    last_message = state["messages"][-1]
    if (hasattr(last_message, "tool_calls") and bool(last_message.tool_calls)):
        tool_call = last_message.tool_calls[-1]
        return tool_call["name"]
    return "END"
            
def build_agent(party:str):
    system_message =  f"You are a legal consultant for {party}.\
                    Speak like how a real legal consultant speaks. Add extra details and insights and talk like a human does.\
                    When asked about the legal sides of the agreements they are provided with \
                    you should use the 'retrive' tool to get the details from the database and answer \
                    the question. When you feel like the knowledge is limited or your knowledge about \
                    something maybe outdated use the 'search_tool' to look up for the latest updates ... \
                    for example about some new ammendments to the laws ...\
                    Whenever there is mention about a legal section or act in the retrieved content, use 'search_tool' to give a brief\
                    explanation of the particular act in the response.\
                    Always try to think creatively and use the 'search_tool' to search\
                    in the internet and fetch data that might be useful for the party.\
                    Always be clear about what you say as your job is sensitive and \
                    for clarity use the tools as much as times you think it might be needed.\
                    Add your suggestions on the content you retrieve from the database.\
                    Always give responses that are explanatory and insightful. It shoud not be too short. Fill the response with the\
                    legal sides of the clause you can find from the internet."
    chatbot_prompt_template = ChatPromptTemplate.from_messages(messages=[("system",system_message),
                                                            MessagesPlaceholder("messages"),])
    graph_builder = StateGraph(State)
    graph_builder.add_node("retrive_node",retrieve_node)
    graph_builder.add_node("search_node",search_node)
    graph_builder.add_node(invoke_llm)

    graph_builder.set_entry_point("invoke_llm")
    graph_builder.add_conditional_edges(
        "invoke_llm",
        conditional_router,
        {
            "END":END,
            "search_tool":"search_node",
            "retrieve":"retrive_node"
        }
    )
    graph_builder.add_edge("search_node","invoke_llm")
    graph_builder.add_edge("retrive_node", "invoke_llm")

    graph = graph_builder.compile(checkpointer = memory)
    return chatbot_prompt_template, graph
