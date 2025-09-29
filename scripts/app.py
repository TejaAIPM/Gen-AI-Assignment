# app_min.py
# one small script to:
# 1) build an index from PDFs/DOCX
# 2) ask a question from that index

import os, json, argparse
import PyPDF2, docx2txt
from sentence_transformers import SentenceTransformer
import faiss

# point to folders
BASE = os.path.dirname(__file__)
SRC  = os.path.abspath(os.path.join(BASE, "..", "Source"))
ART  = os.path.abspath(os.path.join(BASE, "..", "artifacts"))
os.makedirs(ART, exist_ok=True)

# ----- read docs -----
def read_pdf(p):
    # open each PDF and pull text from all pages
    out = []
    with open(p, "rb") as f:
        r = PyPDF2.PdfReader(f)
        for pg in r.pages:
            out.append(pg.extract_text() or "")
    return "\n".join(out)

def read_docx(p):
    # read text from a .docx file
    return docx2txt.process(p) or ""

# ----- chunking -----
def chunk(text, size=800, overlap=100):
    # cut text into overlapping windows
    w = text.split()
    step = max(1, size - overlap)
    return [" ".join(w[i:i+size]) for i in range(0, len(w), step)]

# ----- build index -----
def build_index():
    # collect PDFs/DOCX from Source folder
    docs = []
    for name in sorted(os.listdir(SRC)):
        path = os.path.join(SRC, name)
        if not os.path.isfile(path): continue
        if name.lower().endswith(".pdf"):
            docs.append((name, read_pdf(path)))
        elif name.lower().endswith(".docx"):
            docs.append((name, read_docx(path)))

    # make chunks + metadata
    meta, texts = [], []
    for name, txt in docs:
        if not txt.strip(): continue
        parts = chunk(txt, 800, 100)
        for idx, ch in enumerate(parts):
            meta.append({"source": name, "chunk_index": idx})
            texts.append(ch)

    if not texts:
        print("Nothing to index."); return

    print(f"Indexing {len(texts)} chunks…")
    # turn all chunks into vectors
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    # create a FAISS index (cosine similarity)
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    # save everything
    faiss.write_index(idx, os.path.join(ART, "index.faiss"))
    with open(os.path.join(ART, "store.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "texts": [t[:1000] for t in texts]}, f, ensure_ascii=False)
    print("Saved index.faiss and store.json in artifacts/")

# ----- ask a question -----
def ask(query, top_k=5):
    # load index + texts + meta
    idx  = faiss.read_index(os.path.join(ART, "index.faiss"))
    data = json.load(open(os.path.join(ART, "store.json"), "r", encoding="utf-8"))
    meta, texts = data["meta"], data["texts"]
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # encode query and search FAISS
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = idx.search(q, top_k)
    ids, scores = ids[0].tolist(), scores[0].tolist()

    # show top snippets
    print(f"\nQuestion: {query}\n")
    for rank, (i, s) in enumerate(zip(ids, scores), start=1):
        if i == -1: continue
        m = meta[i]; preview = texts[i].replace("\n", " ")[:300]
        print(f"{rank}. [{m['source']} | chunk {m['chunk_index']}] score={s:.3f}")
        print(f"   {preview}...")
    print()

# ----- CLI -----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd")
    sp.add_parser("build-index")
    q = sp.add_parser("ask")
    q.add_argument("--query", required=True)
    q.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()
    if args.cmd == "build-index": build_index()
    elif args.cmd == "ask": ask(args.query, args.top_k)
    else: ap.print_help()
