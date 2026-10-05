from openai import OpenAI

def format_context(chunks):
    context=""
    for i,chunk in enumerate(chunks,start=1):
        context+=f"[{i}]{chunk['text']}\n"
    return context

system_prompt="""Answer the question using only the context below.
After your answer, list the context numbers you used (e.g. 'Sources: [1], [3]').
If the answer is not in the context , say so."""

def generate_answer(question,chunks):
    client=OpenAI()
    context=format_context(chunks)
    prompt=f"System_prompt:\n{system_prompt}\n\nContext:\n{context}\n\nQuestion:{question}"
    repsonse=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user","content":prompt}]
    )
    answer_text=repsonse.choices[0].message.content
    return answer_text,chunks

def answer_question(questions,chunks):
    return generate_answer(questions,chunks)
