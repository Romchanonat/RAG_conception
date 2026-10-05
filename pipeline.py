import pickle
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage

from Assistant_Python.config import CORPUS_DIR, CHROMA_PERSIST_DIR, OLLAMA_MODEL, DEFAULT_EMBEDDING_MODEL


def load_documents(corpus_dir: str = CORPUS_DIR) -> list:
    loader = DirectoryLoader(
        corpus_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


# 3 stratégies:
#               - small_500: on favorise la précision sémantique au détriment du contexte
#               - medium_1000: on cherche un équilibre entre précision et contexte
#               - token_based: découpage par token afin de maîtriser la taille des chunks par rapport à une limite de tokens
def build_chunking_strategies() -> dict:
    return {
        "small_500": RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
        "medium_1000": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100),
        "token_based": TokenTextSplitter(chunk_size=500, chunk_overlap=50),
    }


def add_semantic_chunker(chunking_strategies: dict, embeddings: HuggingFaceEmbeddings) -> dict:
    chunking_strategies["semantic"] = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",  # autres options: "standard_deviation", "interquartile"
        breakpoint_threshold_amount=95,
    )
    return chunking_strategies


EMBEDDING_MODELS = {
    "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "bge-small-en": "BAAI/bge-small-en-v1.5",
}

CHUNK_CACHE_DIR = Path(".cache_chunks")


def get_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def make_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": get_device()},
        encode_kwargs={"batch_size": 64},
    )


def cached_split(name: str, splitter, documents: list, cache_dir: Path = CHUNK_CACHE_DIR) -> list:
    """Découpe le corpus, ou relit le résultat depuis le cache s'il existe déjà
    (utile pour `semantic`, dont le chunking est lent)."""
    cache_dir.mkdir(exist_ok=True)
    path = cache_dir / f"{name}_{len(documents)}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    chunks = splitter.split_documents(documents)
    path.write_bytes(pickle.dumps(chunks))
    return chunks


def fresh_collection(name: str, emb: HuggingFaceEmbeddings, persist_directory: str = None) -> None:
    """Repart d'une collection vide: sinon une ré-exécution AJOUTE les chunks aux anciens (doublons)."""
    Chroma(collection_name=name, embedding_function=emb, persist_directory=persist_directory).delete_collection()


def build_vectorstore(chunks: list, embeddings: HuggingFaceEmbeddings, collection_name: str,
                       persist_directory: str = None) -> Chroma:
    fresh_collection(collection_name, embeddings, persist_directory)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        collection_metadata={"hnsw:space": "cosine"},  # sinon Chroma utilise la distance L2
        persist_directory=persist_directory,
    )


def build_default_index(documents: list, chunking_strategies: dict) -> tuple:
    embeddings = make_embeddings(DEFAULT_EMBEDDING_MODEL)
    chunks = cached_split("token_based", chunking_strategies["token_based"], documents)
    vectorstore = build_vectorstore(chunks, embeddings, "rag_python", CHROMA_PERSIST_DIR)
    return vectorstore, embeddings, chunks


def make_rag_search(vectorstore: Chroma):
    @tool
    def rag_search(query: str) -> str:
        """Recherche des informations précises dans la documentation Python officielle.
        À utiliser pour toute question sur la syntaxe, les fonctions built-in,
        ou le comportement du langage Python. La requête doit être en anglais."""
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return "Aucun résultat trouvé dans la base de connaissances."
        return "\n\n---\n\n".join([f"[{r.metadata['source']}]\n{r.page_content}" for r in results])

    return rag_search


SYSTEM_PROMPT = """Tu es un assistant expert Python.

Règles :
- Utilise TOUJOURS l'outil rag_search avant de répondre à une question technique sur Python.
- La documentation est en anglais: formule la requête de l'outil en anglais.
- Ne réponds jamais de mémoire seule si l'information peut être vérifiée dans la documentation.
- Si la documentation ne contient pas la réponse, dis-le clairement plutôt que d'inventer (n'invente jamais de module ou de fonction).
- Cite les éléments pertinents trouvés dans ta réponse finale, avec leur source entre crochets.
"""


def build_agent(vectorstore: Chroma, model: str = OLLAMA_MODEL):
    rag_search = make_rag_search(vectorstore)
    llm = ChatOllama(model=model, temperature=0)
    llm_with_tools = llm.bind_tools([rag_search])

    def run_agent(user_question: str, max_iterations: int = 3, history: list = None) -> str:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        
        for question, answer in (history or []):
            messages += [HumanMessage(content=question), AIMessage(content=answer)]
        messages.append(HumanMessage(content=user_question))

        for _ in range(max_iterations):
            ai_message = llm_with_tools.invoke(messages)
            messages.append(ai_message)

            if not ai_message.tool_calls:
                return ai_message.content

            for tool_call in ai_message.tool_calls:
                print(f"[Agent] Appel de l'outil: {tool_call['name']} avec query='{tool_call['args'].get('query')}'")

                if tool_call["name"] == "rag_search":
                    try:
                        result = rag_search.invoke(tool_call["args"])
                    except Exception as e:
                        result = f"Erreur de l'outil: {e}"
                else:
                    result = f"Outil inconnu: {tool_call['name']}"

                messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        
        messages.append(HumanMessage(content="Réponds maintenant avec les informations déjà collectées."))
        return llm.invoke(messages).content

    return llm, run_agent


def chat_history_to_pairs(history: list) -> list:
    """Convertit l'historique Gradio (formats 'messages' ou tuples) en paires (question, réponse) pour run_agent."""
    paires, question = [], None
    for m in history:
        if isinstance(m, dict):
            texte = m["content"] if isinstance(m["content"], str) else m["content"][0]["text"]
            if m["role"] == "user":
                question = texte
            elif question is not None:
                paires.append((question, texte))
        else:
            paires.append(tuple(m))
    return paires
