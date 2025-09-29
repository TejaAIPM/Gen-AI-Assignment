# Conversational Access to Documents – Assignment

This is my quick prototype for the Kanerika Senior AI Engineer assignment.
It lets you ask questions over a folder of mixed files (PDF/DOCX) and get back relevant text snippets.

---

Architecture
- Read PDF/DOCX files from the Source folder
- Split them into overlapping text chunks
- Convert chunks into vectors using sentence transformers
- Store vectors in FAISS for quick similarity search
- When a question is asked, encode the query and retrieve top matching chunks

(Structured CSV/Excel files can be handled with pandas filters omitted here due to time.)

---

How to run

Activate the virtual environment:
source .venv/bin/activate

Build the index from the Source folder:
python scripts/app.py build-index

Ask a question (example):
python scripts/app.py ask --query "What are the benefits of the log book and how can I set it up?" --top_k 5

---

Example query and output

Query:
What are the benefits of the log book and how can I set it up?

Output (top chunk preview):
1. [Doc 5.pdf | chunk 0] score=0.542
SAMPLE User Manual GPS Log Book – ...
2. [Doc 5.pdf | chunk 1] score=0.380
storage capacity • Frequent recording interval ...
...

---

Notes
- Demonstrates conversational access to unstructured docs
- Same approach can be extended to structured data
- Built quickly as a working proof of concept
