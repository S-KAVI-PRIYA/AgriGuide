"""
Crop Recommendation RAG Backend
================================
Pure Python backend (no Streamlit) that other code (CLI, FastAPI, GUI, etc.)
can import and call.

Pipeline:
  CSV rows -> natural-language "documents" -> SentenceTransformer embeddings
  -> ChromaDB vector store -> semantic retrieval -> Gemini generation
  -> optional voice in (Whisper) / voice out (gTTS)

Dataset questions (crops/soil/climate) are answered via RAG using the
Crop_recommendation.csv dataset.

Non-dataset questions are answered using Gemini's built-in Google Search
grounding tool, so the model can pull real, current web information
instead of relying only on its own training data.
"""

import os
import uuid
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# Force CPU so this runs on any machine
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import torch
    torch.cuda.is_available = lambda: False
except ImportError:
    pass

# =====================================================
# 1. CONFIG
# =====================================================

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# DO NOT paste your key here. Set it in your terminal instead, e.g.
# PowerShell (run in the SAME terminal window you launch this script from):
#     $env:GEMINI_API_KEY="your-real-key-from-aistudio"
#     python backend.py
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_PATH = os.environ.get(
    "CROP_CSV_PATH",
    os.path.join(BASE_DIR, "Crop_recommendation.csv")
)
COLLECTION_NAME = "crop_recommendation"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set.\n"
        "In PowerShell run:\n"
        '    $env:GEMINI_API_KEY="your-key-here"\n'
        "then run this script again in the same terminal."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


# =====================================================
# 2. LOAD DATASET -> TEXT DOCUMENTS
# =====================================================

def row_to_document(row: pd.Series) -> str:
    return (
        f"For growing {row['label']}, the ideal conditions are: "
        f"Nitrogen (N) level {row['N']}, Phosphorus (P) level {row['P']}, "
        f"Potassium (K) level {row['K']}, temperature {row['temperature']:.1f}°C, "
        f"humidity {row['humidity']:.1f}%, soil pH {row['ph']:.2f}, "
        f"and rainfall {row['rainfall']:.1f} mm."
    )


def load_documents(csv_path: str = CSV_PATH):
    df = pd.read_csv(csv_path)
    docs = [row_to_document(r) for _, r in df.iterrows()]
    return docs, df


# =====================================================
# 3. EMBEDDING MODEL
# =====================================================

embedding_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")


# =====================================================
# 4. CHROMADB SETUP
# =====================================================

chroma_client = chromadb.PersistentClient(path="./chroma_store")


def get_or_build_collection(rebuild: bool = False):
    """
    Builds the collection once and persists it to disk.
    If the existing collection is empty, rebuild it automatically.
    """

    existing = [c.name for c in chroma_client.list_collections()]

    if rebuild and COLLECTION_NAME in existing:
        chroma_client.delete_collection(COLLECTION_NAME)
        existing.remove(COLLECTION_NAME)

    if COLLECTION_NAME in existing:
        collection = chroma_client.get_collection(COLLECTION_NAME)

        # Rebuild if the existing collection is empty.
        if collection.count() == 0:
            print("Existing ChromaDB collection is empty. Rebuilding...")
            chroma_client.delete_collection(COLLECTION_NAME)
        else:
            return collection

    collection = chroma_client.create_collection(name=COLLECTION_NAME)

    docs, df = load_documents()

    if not docs:
        raise RuntimeError(
            f"No documents were loaded from CSV: {CSV_PATH}"
        )

    store_documents(
        collection,
        docs,
        labels=df["label"].tolist()
    )

    return collection


def store_documents(collection, docs, labels, batch_size=256):
    for start in range(0, len(docs), batch_size):
        batch_docs = docs[start:start + batch_size]
        batch_labels = labels[start:start + batch_size]

        embeddings = embedding_model.encode(
            batch_docs, convert_to_numpy=True
        ).tolist()

        ids = [str(uuid.uuid4()) for _ in batch_docs]
        metadatas = [{"crop": lbl} for lbl in batch_labels]

        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )


# =====================================================
# 5. RETRIEVAL
# =====================================================

def retrieve_context(collection, query: str, top_k: int = 5):
    query_embedding = embedding_model.encode(
        query, convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results["documents"][0]


# =====================================================
# 6. QUESTION ROUTER
# =====================================================

import re

# These are only used as a fast fallback if the AI router fails.
DATASET_KEYWORDS = {
    "crop", "soil", "nitrogen", "phosphorus", "potassium",
    "temperature", "humidity", "rainfall", "grow", "growing",
    "plant", "planting", "farm", "farming", "agriculture", "yield",
    "climate", "season", "cultivat", "irrigation", "fertilizer",
    "fertiliser", "pesticide", "harvest"
}


def keyword_relevance(query: str) -> bool:
    """
    Fast fallback router.
    Returns True when the question looks related to the crop dataset.
    """
    q = query.lower()
    words = set(re.findall(r"[a-z]+", q))

    # Avoid matching very short words accidentally.
    return any(
        word in words or any(word.startswith(kw) for kw in DATASET_KEYWORDS)
        for word in words
    )


def is_relevant(query: str) -> bool:
    """
    AI question router.

    The model decides whether the user's question can reasonably
    be answered using the Crop_recommendation.csv dataset.

    DATASET -> RAG
    GENERAL  -> WEB SEARCH
    """

    router_prompt = f"""
You are a question router for a Crop Recommendation RAG system.

The local dataset contains information about:
- crop names
- nitrogen (N)
- phosphorus (P)
- potassium (K)
- temperature
- humidity
- soil pH
- rainfall
- crop growing conditions
- crop suitability/recommendation based on these conditions

Decide where the question should be answered.

Return ONLY one word:

DATASET
Use DATASET when the question is about crops, soil, farming,
agriculture, crop-growing conditions, crop suitability, or information
that can reasonably be answered from the crop recommendation dataset.

WEB
Use WEB when the question asks for information outside the dataset,
especially current/latest/news/general knowledge/weather/current prices,
people, places, technology, or other unrelated topics.

Important:
- Do not choose DATASET just because the question contains a generic
  word such as "plant".
- If the question is clearly about crop recommendation or crop-growing
  conditions, choose DATASET.
- If current/live information is required, choose WEB.

Question:
{query}

Answer with exactly DATASET or WEB.
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=router_prompt
        )

        decision = response.text.strip().upper()

        if "DATASET" in decision:
            return True

        if "WEB" in decision:
            return False

    except Exception as e:
        print(f"[AI router unavailable: {e}; using keyword fallback]")

    return keyword_relevance(query)


# =====================================================
# 7. ANSWER GENERATION
# =====================================================

def answer_dataset_question(collection, query: str, top_k: int = 5):
    """
    Try to answer from the local RAG dataset.

    Returns:
        (answer, can_answer_from_dataset)
    """
    context_chunks = retrieve_context(collection, query, top_k=top_k)

    if not context_chunks:
        return (
            "The local dataset did not return any relevant information.",
            False
        )

    context = "\n".join(context_chunks)

    prompt = f"""
You are an agricultural assistant backed by a crop recommendation dataset.

Answer the user's question ONLY using the dataset context below.

IMPORTANT:
1. First determine whether the context actually contains enough
   information to answer the question.
2. If it contains enough information, answer clearly and set:
   CAN_ANSWER=YES
3. If it does not contain enough information, do NOT guess. Set:
   CAN_ANSWER=NO
4. If the requested crop is not present in the context, normally set
   CAN_ANSWER=NO.
5. At the end, output exactly:
   CAN_ANSWER=YES
   or
   CAN_ANSWER=NO

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    raw_answer = response.text.strip()

    can_answer = "CAN_ANSWER=YES" in raw_answer.upper()

    # Remove the routing marker from the answer shown to the user.
    clean_answer = re.sub(
        r"\s*CAN_ANSWER\s*=\s*(YES|NO)\s*$",
        "",
        raw_answer,
        flags=re.IGNORECASE
    ).strip()

    return clean_answer, can_answer


def web_search_snippets(query: str, max_results: int = 5):
    """Free, no-API-key web search fallback (used when Gemini quota is hit,
    or can be used as the primary search source). Returns a list of
    {title, body, href} dicts."""
    from ddgs import DDGS
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


def answer_general_question(query: str) -> str:
    """For questions unrelated to the crop dataset.
    Tries Gemini + Google Search grounding first. If that fails
    (e.g. 429 quota exhausted), falls back to a free DuckDuckGo web
    search and summarizes/returns the raw results instead."""

    try:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        prompt = (
            "Answer the following question helpfully and concisely, "
            "using web search if it helps you give an accurate, "
            "up-to-date answer.\n\n"
            f"Question: {query}"
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config,
        )

        source_note = "\n\n[Source: WEB - Gemini Google Search grounding]"
        return response.text + source_note

    except Exception as e:
        # Quota hit, network issue, etc. -> fall back to free DDG search.
        print(f"[Gemini web search unavailable ({e}); falling back to DuckDuckGo search]")
        try:
            results = web_search_snippets(query, max_results=5)
        except Exception as ddg_err:
            return (
                f"Sorry, I couldn't reach any search backend right now "
                f"(Gemini error: {e}; DuckDuckGo error: {ddg_err})."
            )

        if not results:
            return "No web search results found for that question."

        lines = [f"Here's what I found on the web for: \"{query}\"\n"]
        for r in results:
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            href = r.get("href", "").strip()
            lines.append(f"• {title}\n  {body}\n  {href}\n")

        source_note = "\n[Source: WEB - DuckDuckGo search fallback]"
        return "\n".join(lines) + source_note


def answer_question(collection, query: str) -> str:
    """
    Main question router.

    DATASET question -> RAG
    If RAG cannot answer -> Web Search
    GENERAL question -> Web Search
    """
    if not query.strip():
        return "Please enter a question."

    use_rag = is_relevant(query)

    if use_rag:
        print("[Router: DATASET -> RAG]")

        rag_answer, can_answer = answer_dataset_question(
            collection,
            query
        )

        if can_answer:
            return (
                rag_answer
                + "\n\n[Source: RAG - local Crop_recommendation.csv dataset]"
            )

        print("[RAG: Dataset does not contain enough information]")
        print("[Fallback: WEB -> Web Search]")

        web_answer = answer_general_question(query)

        return (
            web_answer
            + "\n\n[Note: RAG could not answer from the local dataset, "
              "so web search was used.]"
        )

    print("[Router: WEB -> Web Search]")
    return answer_general_question(query)


# =====================================================
# 8. VOICE I/O (Whisper in, gTTS out)
# =====================================================

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    import whisper
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)
    return result["text"].strip()


def speak_answer(text: str, out_path: str = "answer.mp3", lang: str = "en"):
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    return out_path


# =====================================================
# 9. CLI ENTRYPOINT
# =====================================================

def main():
    print("Loading dataset and building/loading vector store...")
    collection = get_or_build_collection()
    print(f"Ready. {collection.count()} records indexed.\n")
    print("Type 'exit' to quit. Type 'voice' to ask by microphone recording file.\n")

    while True:
        query = input("Ask: ").strip()

        if query.lower() == "exit":
            print("Goodbye.")
            break

        if query.lower() == "voice":
            audio_path = input("Path to audio file: ").strip()
            try:
                query = transcribe_audio(audio_path)
                print(f"Transcribed: {query}")
            except Exception as e:
                print(f"Transcription error: {e}")
                continue

        if not query:
            continue

        try:
            answer = answer_question(collection, query)
            print(f"\nAnswer:\n{answer}\n")

            speak = input("Speak this answer aloud (y/n)? ").strip().lower()
            if speak == "y":
                path = speak_answer(answer)
                print(f"Saved audio to {path}\n")

        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()