import sys
from ingest import ingest_pdf
from retrieve import retrieve_chunks
from answer import answer_question

command = sys.argv[1]

if command == "ingest":
    pdf_path = sys.argv[2]
    ingest_pdf(pdf_path)

elif command == "ask":
    question = sys.argv[2]
    chunks = retrieve_chunks(question, n_results=3)
    answer, cited_chunks = answer_question(question, chunks)
    print(f"\nAnswer:\n{answer}\n")
    print("Retrieved passages:")
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}] {chunk['text'][:120]}... (from {chunk['source']})")