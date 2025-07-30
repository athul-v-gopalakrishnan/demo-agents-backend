
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain_core.messages import HumanMessage, AIMessage

from typing_extensions import List, AsyncGenerator
from pydantic import BaseModel

from rag_chatbot.agent import build_agent as build_rag_agent
from rag_chatbot.embed_data import load_uploaded_pdfs as load_rag_documents \
                                        , embed_docs as embed_rag_docs \
                                        , clear_all_pgvector_data as clear_rag_data


from document_analyzer.agent import build_agent as build_agreement_agent \
                                        , analyze_document as analyze_agreement
from document_analyzer.embed_data import load_uploaded_pdfs as load_agreement \
                                        , embed_docs as embed_agreements

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=['https://athul-v-gopalakrishnan.github.io/demo-agents-frontend/', 
                   'https://athul-v-gopalakrishnan.github.io/demo-agents-frontend/rag_chatbot/main.html',
                   'https://athul-v-gopalakrishnan.github.io/demo-agents-frontend/document_analyzer/main.html',
                   'https://athul-v-gopalakrishnan.github.io/demo-agents-frontend/document_analyzer/analyze.html'], 
    allow_credentials=True,
    allow_methods=["GET", "POST"],     # Add more if needed
    allow_headers=["Content-Type", "Authorization"],
)

class RagRequestModel(BaseModel):
    question:str
    temperature:float
    max_retrievals:int

class RagResponseModel(BaseModel):
    response:str

class AgreementRequestModel(BaseModel):
    question:str

session_memory = {
    "agreement_agent":None,
    "template":None,
    "summary":None,
}

rag_agent = build_rag_agent()

# =======RAG APIs==========

@app.post("/rag_chatbot/upload")
async def upload_rag_docs(files:List[UploadFile] = File(...)):
    docs = await load_rag_documents(files)
    rag_vector_store = embed_rag_docs(docs)
    return {"message" : "Document uploaded succesfully."}

@app.post("/rag_chatbot/query")
async def stream_rag_query(req: RagRequestModel):
    current_state = {
        "messages": [HumanMessage(content=req.question)],
        "temperature": req.temperature,
        "num_results": req.max_retrievals
    }

    async def rag_token_stream() -> AsyncGenerator[str, None]:
        async for step in rag_agent.astream(
            current_state,
            config={
                "configurable": {
                    "thread_id": "user123",
                    "temperature": req.temperature,
                    "num_results": req.max_retrievals
                }
            },
            stream_mode="values"
        ):
            msg = step["messages"][-1]
            if isinstance(msg, AIMessage):
                yield msg.content

    return StreamingResponse(rag_token_stream(), media_type="text/plain")

@app.post("/rag_chatbot/clear_db")
def clear_db():
    global rag_agent
    clear_rag_data()
    rag_agent = build_rag_agent()
    return {"response" : "Database Cleared."}


# ================Agreement Analyzer APIs=========================

@app.post("/analyzer/upload")
async def upload_agreement(file:UploadFile = File(...), party: str = Form(...)):
    docs = await load_agreement(file)
    embed_agreements(docs)
    template, agent = build_agreement_agent(party)
    session_memory["agent"] = agent
    session_memory["template"] = template
    document_content = docs[0].page_content
    summary = analyze_agreement(document_content, party)
    session_memory["summary"] = summary
    return {"done" : True}

@app.get("/analyzer/generate_summary")
async def retrieve_summary():
    return session_memory["summary"]

@app.post("/analyzer/chat")
async def stream_agreement_query(req:AgreementRequestModel):
    current_state = {
        "messages": [HumanMessage(content=req.question)],
    }

    async def token_stream() -> AsyncGenerator[str, None]:
        async for step in session_memory['agent'].astream(
            current_state,
            config={
                "configurable": {
                    "thread_id": "user123"
                }
            },
            stream_mode="values"
        ):
            msg = step["messages"][-1]
            if isinstance(msg, AIMessage):
                yield msg.content

    return StreamingResponse(token_stream(), media_type="text/plain")