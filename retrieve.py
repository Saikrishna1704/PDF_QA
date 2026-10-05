from openai import OpenAI
import chromadb

def embed_question(question):
    client=OpenAI()
    response=client.embeddings.create(
        model="text-embeddings-3-small",
        input=[question]
    )
    embedding=response.data[0].embedding
    return embedding

def query_chroma(question_embedding,n_results):
    chroma_client=chromadb.Client()
    collection=chroma_client.get_collection("pdf_qa")
    results=collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )
    output=[]
    for doc,meta in zip(results["documents"][0],results["metadata"][0]):
        output.append({
            "text":doc,
            "chunk_index":meta.get("chunk_index")
        })
    return output

def retrieve_chunks(question,n_results):
    que_embed=embed_question(question)
    output=query_chroma(que_embed,n_results)
    return output