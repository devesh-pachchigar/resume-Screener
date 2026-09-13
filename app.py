import os
import tempfile

import streamlit as st

from resume_engine import (
    build_vectorstore,
    build_combined_store,
    evaluate_resume,
    compare_candidates,
    build_qa_chain
)

st.set_page_config(page_title="AI Resume Screening Assistant", layout="wide")
st.title("AI Resume Screening Assistant (LangChain + RAG)")


# ---------------------------------------------------------
# 1. Sidebar -> upload resumes + job description
# ---------------------------------------------------------
with st.sidebar:
    st.header("Step 1: Upload Resumes")

    uploaded_files = st.file_uploader(
        "Upload one or more resumes (PDF only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.header("Step 2: Job Description")

    job_description = st.text_area(
        "Paste the Job Description here",
        height=250,
        value=(
            "Role: Data Scientist\n"
            "Must have: Python, SQL, Machine Learning, Scikit-learn, "
            "Statistics, Pandas, Data Visualization\n"
            "Good to have: Deep Learning, NLP, LangChain, AWS, MLOps\n"
            "Experience: 3+ years in building and deploying ML models."
        )
    )

    process = st.button("Process Resumes")

    st.markdown("---")
    st.info("Everything is answered only from the uploaded resumes (RAG).")


# ---------------------------------------------------------
# 2. Process the uploaded resumes (build the vector DBs)
# ---------------------------------------------------------
if process:

    if not uploaded_files:
        st.warning("Please upload at least one resume PDF.")
    elif not job_description.strip():
        st.warning("Please enter a Job Description.")
    else:
        with st.spinner("Reading resumes and creating the vector database..."):

            resume_stores = {}   # {candidate_name: FAISS store}
            all_docs = []        # all chunks of all resumes (for the Q&A tab)

            for pdf_file in uploaded_files:

                # Streamlit gives us an in-memory file, so write it to disk
                # first, because PyPDFLoader needs a real file path.
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, pdf_file.name)

                with open(file_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                candidate_name = pdf_file.name.replace(".pdf", "")

                vs, docs = build_vectorstore(file_path, candidate_name)

                resume_stores[candidate_name] = vs
                all_docs.extend(docs)

            # One combined FAISS store, used by the Q&A chatbot tab
            combined_store = build_combined_store(all_docs)

            # Save everything in session_state so it survives re-runs
            st.session_state.resume_stores = resume_stores
            st.session_state.combined_store = combined_store
            st.session_state.job_description = job_description
            st.session_state.evaluations = {}

        st.success(f"{len(resume_stores)} resume(s) processed successfully.")


# ---------------------------------------------------------
# 3. Main area -> 3 tabs
# ---------------------------------------------------------
if "resume_stores" in st.session_state:

    tab1, tab2, tab3 = st.tabs(
        ["1. Evaluate Resumes", "2. Compare Candidates", "3. Ask Questions"]
    )

    # -----------------------------------------------------
    # TAB 1 -> Evaluate every resume against the JD
    # -----------------------------------------------------
    with tab1:
        st.subheader("Candidate Evaluation")

        if st.button("Evaluate All Resumes"):

            for name, store in st.session_state.resume_stores.items():

                with st.spinner(f"Evaluating {name}..."):

                    result = evaluate_resume(
                        store,
                        st.session_state.job_description,
                        name
                    )

                    st.session_state.evaluations[name] = result

        # ---- Show the evaluation results ----
        for name, result in st.session_state.get("evaluations", {}).items():

            st.markdown(f"### {name}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Match Score", f"{result['match_score']} / 100")
            col2.metric("Recommendation", result["recommendation"])
            col3.metric("Missing Skills", len(result["missing_skills"]))

            st.write("**Candidate Summary:**", result["candidate_summary"])

            c1, c2 = st.columns(2)

            with c1:
                st.success("Matching Skills")
                for s in result["matching_skills"]:
                    st.write("-", s)

                st.info("Strengths")
                for s in result["strengths"]:
                    st.write("-", s)

            with c2:
                st.error("Missing Skills")
                for s in result["missing_skills"]:
                    st.write("-", s)

                st.warning("Weaknesses")
                for s in result["weaknesses"]:
                    st.write("-", s)

            st.write("**Justification:**", result["justification"])
            st.markdown("---")

    # -----------------------------------------------------
    # TAB 2 -> Compare candidates / pick the best one
    # -----------------------------------------------------
    with tab2:
        st.subheader("Compare Candidates")

        if not st.session_state.get("evaluations"):
            st.info("Run the evaluation in Tab 1 first.")
        else:
            # Simple score table
            table = []
            for name, r in st.session_state.evaluations.items():
                table.append({
                    "Candidate": name,
                    "Match Score": r["match_score"],
                    "Recommendation": r["recommendation"],
                    "Missing Skills": ", ".join(r["missing_skills"][:4])
                })

            table = sorted(table, key=lambda x: x["Match Score"], reverse=True)
            st.dataframe(table, use_container_width=True)

            if st.button("Recommend the Best Candidate"):

                with st.spinner("Comparing candidates..."):

                    answer = compare_candidates(
                        st.session_state.evaluations,
                        st.session_state.job_description
                    )

                st.markdown(answer)

    # -----------------------------------------------------
    # TAB 3 -> RAG chatbot over all resumes
    # -----------------------------------------------------
    with tab3:
        st.subheader("Ask Questions About the Resumes")

        rag_chain = build_qa_chain(st.session_state.combined_store)

        query = st.text_input(
            "Example: Which candidate has AWS experience?"
        )

        if query:
            with st.spinner("Searching the resumes..."):
                response = rag_chain.invoke(query)
            st.markdown(response)

else:
    st.info("Upload resumes and a Job Description in the sidebar, then click 'Process Resumes'.")
