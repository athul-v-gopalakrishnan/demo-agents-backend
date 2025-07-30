from dotenv import load_dotenv
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_chatbot.embed_data import rag_vector_store

from langchain_openai import OpenAIEmbeddings

from db_config import *

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

async def load_uploaded_pdfs(uploaded_file):
    docs = ""
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await uploaded_file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    loader = PyPDFLoader(tmp_path)
    doc = loader.load()
    return doc


def embed_docs(docs):
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
#     dbname=dbname,
#     user=user,
#     password=password,
#     host=host,
#     port=port
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

if __name__ == "__main__":
    clear_all_pgvector_data()   

    
