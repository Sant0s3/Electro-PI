# Section 2: Small RAG System

Retrieval-augmented generation pipeline built with LangChain, FAISS, and LangGraph. Reads a PDF, chunks it, indexes the chunks, retrieves relevant passages per question, and generates an answer using Groq's Llama 3.3 70B model.

## How It Works

1. Loads `pdf.pdf` (an ensemble learning lecture) and splits it into overlapping chunks using `RecursiveCharacterTextSplitter`.
2. Embeds the chunks with `all-MiniLM-L6-v2` (local, no API key needed) and stores them in a FAISS index.
3. For each question, retrieves the top 3 most relevant chunks by cosine similarity.
4. Builds a prompt with the retrieved context and sends it to Groq (Llama 3.3 70B) for answer generation.
5. If the context doesn't contain the answer, the model says so instead of guessing.

## Setup

```bash
cd "Small RAG"
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key:
```env
GROQ_API_KEY=your-groq-key
```

## Run

```bash
python agent.py
```

## Example Output

```
Q: What is boosting in ensemble learning?
A: Boosting is an ensemble method that builds trees sequentially, with each
   new tree correcting the errors of previous trees. It consists of weak
   learners and a sequential additive model. [Chunk 0, Chunk 16]

Q: What is the capital of Egypt?
A: I cannot find the answer in the provided context. [Chunk 22]
```

The last question is intentionally out-of-scope to show that the pipeline refuses to hallucinate when the PDF doesn't contain the answer.

## Write-up: Improving Retrieval Quality

If the pipeline struggles on longer or denser documents, a few things would help:

- **Overlapping chunks**: Right now we use a 200-char overlap. For technical PDFs, bumping this up or switching to semantic chunking (splitting on paragraph/section boundaries) catches more context at chunk edges.
- **Hybrid search**: Combine the embedding-based similarity search with a keyword search (BM25). This catches cases where the question uses exact terminology that embeddings might not weight highly enough.
- **Re-ranking**: After retrieving the top-K chunks, run a cross-encoder re-ranker (like `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score them. The initial retrieval casts a wide net, the re-ranker tightens it.
- **Chunk size tuning**: Keeping chunks moderate (500-1000 chars) and preserving document structure (headings, lists) helps the model understand what each chunk is about without losing surrounding context.
