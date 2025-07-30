from dotenv import load_dotenv
from typing import Sequence
from typing_extensions import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage,BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from db_config import DB_CONNECTION_STRING
from rag_chatbot.embed_data import rag_vector_store

load_dotenv()

class State(TypedDict):
    messages:Annotated[Sequence[BaseMessage],add_messages]
    # doc:str
    temperature:float

search = TavilySearch(max_results = 3)
search.name = "search_tool"
search_node = ToolNode(tools=[search], name= "search_node")

llm = init_chat_model("openai:gpt-4.1")
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
memory = MemorySaver()
prompt_template = ChatPromptTemplate.from_template(
    "You are a chatbot assistant. You can search the web if needed using the 'search_node'. Use the tool 'retrieve' when the information is insufficient.{query}"
)

# def check_for_tool_calls(state:State):
#     last_message = state["messages"][-1]
#     return (hasattr(last_message, "tool_calls") and bool(last_message.tool_calls))

@tool(response_format="content_and_artifact", description="Retrieve data from vector store")
def retrieve(
    query:str, 
    config:RunnableConfig
    ):
    num_results = config.get("configurable",{}).get("num_results",8)
    # retriever = PGVector(
    #         embeddings=embedding_model,
    #         collection_name="embeddings",
    #         connection=DB_CONNECTION_STRING
    #     )
    retriever = rag_vector_store
    retrieved_docs = retriever.similarity_search(query = query, k = num_results)
    serialised = "\n\n".join(
        (f"Source : {doc.metadata}\nContent:{doc.page_content}")
        for doc in retrieved_docs
    )
    return serialised, retrieved_docs
    
retrieve_node = ToolNode([retrieve], name = "retrieve_node")
llm_with_tools = llm.bind_tools(tools = [retrieve,search], tool_choice="auto")

def conditional_router(state:State):
    last_message = state["messages"][-1]
    if (hasattr(last_message, "tool_calls") and bool(last_message.tool_calls)):
        tool_call = last_message.tool_calls[-1]
        return tool_call["name"]
    return "END"

def query_or_respond(state:State):
    temperature = state.get("temperature",0.7)
    prompt = prompt_template.invoke({
        "query":state["messages"]
    })
    response = llm_with_tools.invoke(prompt,config={"temperature":temperature})
    return {"messages":[response]}

def generate(state:State):
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    tool_messages = recent_tool_messages[::-1]

    search_results = "\n\n".join(result.content for result in tool_messages)

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{search_results}"
    )

    conversational_messages = [
        message 
        for message in state["messages"]
        if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversational_messages
    response = llm.invoke(prompt,config={"temperature":state.get("temperature",0.7)})

    return {"messages" : [response]}

def build_agent():
    graph_builder = StateGraph(State)
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(search_node)
    graph_builder.add_node(retrieve_node)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        conditional_router,
        {
            "END":END,
            "search_tool":"search_node",
            "retrieve":"retrieve_node"
        }
    )
    graph_builder.add_edge("retrieve_node","generate")
    graph_builder.add_edge("search_node","generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile(checkpointer = memory)
    return graph