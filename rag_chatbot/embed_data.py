from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

from fastapi import UploadFile
from typing_extensions import List

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_openai import OpenAIEmbeddings

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
rag_vector_store = InMemoryVectorStore(embedding_model)

async def load_uploaded_pdfs(uploaded_files:List[UploadFile]):
    docs = []
    for uploaded_file in uploaded_files:
        content = await uploaded_file.read()
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        loader = PyPDFLoader(tmp_path)
        doc = loader.load()
        docs+=doc
    return docs


def embed_docs(docs):
    rag_vector_store.delete()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, add_start_index=True, separators=["\n", ".", "!", "?", ",", " "]
    )
    all_splits = text_splitter.split_documents(docs)

    # PGVector.from_documents(
    #     documents=all_splits,
    #     embedding=embedding_model,
    #     collection_name="embeddings",
    #     connection=DB_CONNECTION_STRING
    #     )
    rag_vector_store.add_documents(documents=all_splits)

    

def clear_all_pgvector_data():
#     conn = psycopg2.connect(
#     dbname="rag_data",
#     user="demo_rag",
#     password="rag_pass",
#     host="localhost",
#     port=5432
#     )
#     cur = conn.cursor()

#     # Clear one collection
#     collection_name = "embeddings"
#     cur.execute("""
#     DELETE FROM langchain_pg_embedding
#     WHERE collection_id = (
#         SELECT id FROM langchain_pg_collection
#         WHERE name = %s
#     )::uuid;
# """, (collection_name,))

#     cur.execute("""
#         DELETE FROM langchain_pg_collection
#         WHERE name = %s;
#     """, (collection_name,))

#     conn.commit()
#     cur.close()
#     conn.close()
    rag_vector_store.delete()

# clear_all_pgvector_data()


    

    
