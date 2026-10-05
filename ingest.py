import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb


def load_pdf(pdf_path):
    doc=fitz.open(pdf_path)
    text_pages=[]
    for page_num in range(len(doc)):
        page=doc[page_num]
        text=page.get_text()
        text_pages.append(f"--- Page {page_num+1} ---\n{text}")
    return "\n".join(text_pages)

def split_text(text):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80
    )
    chunks=splitter.split_text(text)
    return chunks

def embed_chunks(chunks):
    client=OpenAI()
    response=client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    embeddings=[item.embedding for item in response.data]
    return embeddings

def store_in_chroma(chunks,embeddings,pdf_path):
    chroma_client=chromadb.Client()
    collection=chroma_client.get_or_create_collection("pdf_qa")
    for i,(chunk,embedding) in enumerate(zip(chunks,embeddings)):
        collection.add(
            ids=[f"{pdf_path}_chunk_{i}"],
            embeddings=[embedding],
            document=[chunk],
            metadata=[{"chunk_index":i}]
        )
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")

def injest_pdf(pdf_path):
    text_pages=load_pdf(pdf_path)
    chunks=split_text(text_pages)
    embeddings=embed_chunks(chunks)
    store_in_chroma(chunks,embeddings,pdf_path)

