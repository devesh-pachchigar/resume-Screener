# =========================================================
# Q1. RESUME SCREENING -> RAG ENGINE
# =========================================================
# This file has ALL the LangChain logic and no Streamlit code.
# app.py only imports from here and draws the screen.
#
# Pipeline:
#   PDF Loader -> Text Splitter -> Embeddings -> FAISS
#   -> Retriever -> Prompt Template -> LLM -> Output Parser
# =========================================================

from dotenv import load_dotenv

# ---- LangChain imports (only the ones used in class) ----
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# ---------------------------------------------------------
# 1. Load the API key from the .env file
# ---------------------------------------------------------
load_dotenv()   # .env must contain: OPENAI_API_KEY=sk-xxxx


# ---------------------------------------------------------
# 2. Models (LLM + Embeddings)
# ---------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# ---------------------------------------------------------
# 3. Prompt templates
# ---------------------------------------------------------

# 3.1 Evaluation prompt -> returns JSON so we can parse it
evaluation_prompt = ChatPromptTemplate.from_template("""
You are an experienced technical recruiter.

Evaluate the candidate ONLY using the resume context given below.
If something is not present in the context, treat it as missing.
Do NOT invent any experience, skill or qualification.

JOB DESCRIPTION:
{jd}

RESUME CONTEXT:
{context}

Return ONLY valid JSON in exactly this format (no extra text):
{{
 "candidate_name": "",
 "match_score": 0,
 "matching_skills": [],
 "missing_skills": [],
 "candidate_summary": "",
 "strengths": [],
 "weaknesses": [],
 "recommendation": "",
 "justification": ""
}}

Rules:
- "match_score" is a number between 0 and 100.
- "recommendation" must be one of: "Strong Fit", "Moderate Fit", "Not a Fit".
- Keep every list item short (maximum 8 words).
- "justification" must be 2 to 3 lines explaining the score.
""")

# 3.2 Comparison prompt -> plain text answer
comparison_prompt = ChatPromptTemplate.from_template("""
You are a hiring manager.

Below are the evaluation results of all candidates for the same job.

JOB DESCRIPTION:
{jd}

CANDIDATE EVALUATIONS:
{evaluations}

Give your answer in this format:
1. Best Candidate: <name>
2. Why this candidate is the best: 3 bullet points
3. Ranking of all candidates with their scores
4. Final hiring recommendation in 2 lines
""")

# 3.3 Q&A prompt -> strict RAG, answers only from resumes
qa_prompt = ChatPromptTemplate.from_template("""
You are a resume screening assistant.

Answer the question using ONLY the resume context below.
If the answer is not present in the context, reply exactly:
"This information is not available in the uploaded resumes."

Context:
{context}

Question: {question}

Answer in short bullet points.
""")


# ---------------------------------------------------------
# 4. Helper functions
# ---------------------------------------------------------
def format_docs(docs):
    """Join the retrieved chunks into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_vectorstore(file_path, candidate_name):
    """
    Takes ONE resume PDF (file path) and returns:
      - a FAISS vector store for that resume
      - the list of chunks (used later for the combined store)

    Steps: PyPDFLoader -> split -> tag -> embed -> FAISS
    """

    # 4.1 Load the PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 4.2 Chunking (same settings as the class RAG project)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    # 4.3 Tag every chunk with the candidate name.
    #     This is what tells us which resume a chunk came from.
    for d in docs:
        d.metadata["candidate"] = candidate_name

    # 4.4 Embeddings + FAISS vector DB
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore, docs


def build_combined_store(all_docs):
    """One FAISS store holding every resume -> used by the Q&A tab."""
    return FAISS.from_documents(all_docs, embeddings)


# ---------------------------------------------------------
# 5. Chain 1 -> Evaluate one resume against the JD
# ---------------------------------------------------------
def evaluate_resume(vectorstore, job_description, candidate_name):
    """
    Retrieves the resume chunks closest to the JD and asks the LLM
    for a structured evaluation (JSON).
    """

    # 5.1 Retriever pulls the 4 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(job_description)
    context = format_docs(docs)

    # 5.2 Chain: prompt -> llm -> JSON output parser
    evaluation_chain = evaluation_prompt | llm | JsonOutputParser()

    result = evaluation_chain.invoke({
        "jd": job_description,
        "context": context
    })

    # 5.3 Always keep the file name as the candidate name
    result["candidate_name"] = candidate_name

    return result


# ---------------------------------------------------------
# 6. Chain 2 -> Compare all candidates
# ---------------------------------------------------------
def compare_candidates(evaluations, job_description):
    """Takes all evaluation results and recommends the best candidate."""

    comparison_chain = comparison_prompt | llm | StrOutputParser()

    return comparison_chain.invoke({
        "jd": job_description,
        "evaluations": str(evaluations)
    })


# ---------------------------------------------------------
# 7. Chain 3 -> RAG chatbot over all resumes
# ---------------------------------------------------------
def build_qa_chain(combined_store):
    """
    Classic RAG chain (same structure as the YouTube Assistant project):
    retriever -> prompt -> llm -> string output parser
    """

    retriever = combined_store.as_retriever(search_kwargs={"k": 4})

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
