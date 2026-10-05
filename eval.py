import re
import time

import numpy as np
import pandas as pd

from pipeline import build_vectorstore, cached_split, make_embeddings

golden_set = [
    {
        "question": "Comment fonctionne une liste en compréhension en Python ?",
        "expected_keywords": ["comprehension", "compréhension", "list"],
    },
    {
        "question": "Comment définir une fonction avec des arguments par défaut ?",
        "expected_keywords": ["def", "default", "argument"],
    },
    {
        "question": "Quelle est la différence entre une liste et un tuple ?",
        "expected_keywords": ["tuple", "list", "immutable"],
    },
    {
        "question": "Comment gérer les exceptions avec try/except ?",
        "expected_keywords": ["try", "except", "exception"],
    },
    {
        "question": "Comment ouvrir et lire un fichier en Python ?",
        "expected_keywords": ["open", "file", "read"],
    },
    {
        "question": "Comment utiliser itertools.groupby pour grouper des éléments ?",
        "expected_keywords": ["itertools", "groupby", "iterable"],
    },
    {
        "question": "Comment manipuler des chemins de fichiers avec pathlib ?",
        "expected_keywords": ["pathlib", "Path", "joinpath"],
    },
    {
        "question": "Comment sérialiser et désérialiser du JSON en Python ?",
        "expected_keywords": ["json", "dumps", "loads"],
    },
    {
        "question": "Comment créer une fonction génératrice avec yield ?",
        "expected_keywords": ["yield", "generator", "iterator"],
    },
    {
        "question": "Comment fusionner deux dictionnaires avec l'opérateur de fusion ?",
        "expected_keywords": ["dict", "merge", "update"],
    },
    {
        "question": "Comment utiliser des fonctions anonymes lambda ?",
        "expected_keywords": ["lambda", "anonymous", "function"],
    },
    {
        "question": "Comment parcourir une liste avec son index en même temps ?",
        "expected_keywords": ["enumerate", "index", "loop"],
    },
    {
        "question": "Comment utiliser le module collections pour compter des éléments ?",
        "expected_keywords": ["collections", "Counter", "frequency"],
    },
    {
        "question": "Comment définir une structure de données simple avec dataclasses ?",
        "expected_keywords": ["dataclass", "field", "decorator"],
    },
    {
        "question": "Comment valider des expressions régulières avec le module re ?",
        "expected_keywords": ["re", "match", "search", "pattern"],
    },
    {
        "question": "Comment gérer des tâches asynchrones avec asyncio ?",
        "expected_keywords": ["asyncio", "async", "await", "gather"],
    },
    {
        "question": "Comment ajouter des annotations de type (type hints) ?",
        "expected_keywords": ["typing", "Optional", "Union", "int"],
    },
    {
        "question": "Comment parser des arguments en ligne de commande avec argparse ?",
        "expected_keywords": ["argparse", "ArgumentParser", "parse_args"],
    },
    {
        "question": "Comment configurer le module de journalisation logging ?",
        "expected_keywords": ["logging", "basicConfig", "level", "debug"],
    },
    {
        "question": "Comment utiliser l'opérateur morse (walrus operator) ?",
        "expected_keywords": ["assignment", "expression", "walrus"],
    },
    {
        "question": "Quelle est la recette de la tarte aux pommes traditionnelle ?",
        "expected_keywords": ["désolé", "pas trouvé", "hors sujet", "documentation Python"],
        "should_fail": True,
    },
    {
        "question": "Comment utiliser le module natif 'quantum_compute' introduit en Python 3.15 ?",
        "expected_keywords": ["n'existe pas", "pas dans", "inconnu"],
        "should_fail": True,
    },
    {
        "question": "Comment configurer un serveur web avec le framework intégré 'py_express' ?",
        "expected_keywords": ["pas partie", "bibliothèque standard", "externe"],
        "should_fail": True,
    },
    {
        "question": "Quelle est la syntaxe exacte de la fonction `print_colorful()` dans la bibliothèque standard ?",
        "expected_keywords": ["pas", "existe", "standard"],
        "should_fail": True,
    },
    {
        "question": "Comment coder un smart contract Solidity directement en Python natif ?",
        "expected_keywords": ["Solidity", "Ethereum", "pas supporté", "hors périmètre"],
        "should_fail": True,
    },
]


def is_relevant(chunk_text: str, expected_keywords: list) -> bool:
    return any(re.search(rf"(?<!\w){re.escape(kw)}(?:s|es)?(?!\w)", chunk_text, re.IGNORECASE) for kw in expected_keywords)


def evaluate_retrieval(vectorstore, golden_set: list, k: int = 3) -> dict:
    """Calcule recall@k, MRR et latence moyenne pour un vectorstore donné."""
    hits, reciprocal_ranks, latencies, sim_scores = [], [], [], []
    vectorstore.similarity_search("warm-up", k=1) 

    for item in golden_set:
        if item.get("should_fail"):
            continue
        start = time.time()
        results = vectorstore.similarity_search_with_score(item["question"], k=k)
        latencies.append(time.time() - start)

        rank = None
        for i, (doc, score) in enumerate(results, start=1):
            sim_scores.append(1 - score)
            if rank is None and is_relevant(doc.page_content, item["expected_keywords"]):
                rank = i

        hits.append(1 if rank is not None else 0)
        reciprocal_ranks.append(1 / rank if rank else 0)

    return {
        f"Recall@{k}": round(np.mean(hits), 3),
        "MRR": round(np.mean(reciprocal_ranks), 3),
        "Score similarité moyen": round(np.mean(sim_scores), 4) if sim_scores else None,
        "Latence retrieval moyenne (s)": round(np.mean(latencies), 4),
    }


def run_comparison(documents: list, chunking_strategies: dict, embedding_models: dict, golden_set: list,
                    k: int = 3) -> pd.DataFrame:
    all_results = []
    embeddings_cache = {}

    for chunk_name, splitter in chunking_strategies.items():
        t0 = time.time()
        chunks = cached_split(chunk_name, splitter, documents)
        chunk_time = time.time() - t0

        for emb_name, emb_model_name in embedding_models.items():
            print(f"--- Combo: chunking={chunk_name} | embedding={emb_name} ---")
            try:
                if emb_name not in embeddings_cache:
                    embeddings_cache[emb_name] = make_embeddings(emb_model_name)
                emb = embeddings_cache[emb_name]

                t0 = time.time()
                
                collection = f"{chunk_name}_{emb_name}"
                vs = build_vectorstore(chunks, emb, collection)
                index_time = time.time() - t0

                metrics = evaluate_retrieval(vs, golden_set, k=k)

                tok = emb._client.tokenizer
                lengths = [len(tok(c.page_content, truncation=False, verbose=False)["input_ids"]) for c in chunks[::20]]
                tronques = round(100 * np.mean(np.array(lengths) > emb._client.max_seq_length), 1)

                all_results.append({
                    "chunking": chunk_name,
                    "embedding": emb_name,
                    "nb_chunks": len(chunks),
                    "longueur_moyenne": int(sum(len(c.page_content) for c in chunks) / len(chunks)),
                    "temps_chunking (s)": round(chunk_time, 4),
                    "temps_indexation (s)": round(index_time, 4),
                    "tronqués (%)": tronques,
                    **metrics,
                })
                vs.delete_collection()
            except Exception as e:
                print(f"Erreur sur la combo {chunk_name}/{emb_name}: {e}")

    return pd.DataFrame(all_results)


def score_answer(answer: str, expected_keywords: list, question: str = "") -> int:
    # on ignore les mots-clés déjà présents dans la question: le LLM les répète, ça ne prouve rien
    keywords = [kw for kw in expected_keywords if kw.lower() not in question.lower()] or expected_keywords
    return int(is_relevant(answer, keywords))


def is_refusal(question: str, answer: str, llm) -> int:
    """Questions hors périmètre: l'assistant a-t-il refusé au lieu d'inventer ? (LLM-juge, même modèle que l'agent)"""
    prompt = (f"Question posée : {question}\n\nRéponse d'un assistant : {answer}\n\n"
              "L'assistant a-t-il reconnu que l'information n'est pas disponible dans la documentation Python, "
              "que le sujet est hors périmètre ou que l'élément demandé n'existe pas, au lieu d'inventer une réponse ? "
              "Réponds uniquement par OUI ou NON.")
    return int(llm.invoke(prompt).content.strip().upper().startswith("OUI"))


def evaluate_agent_e2e(golden_set: list, run_agent_fn, llm) -> pd.DataFrame:
    """Rejoue l'agent complet sur le golden set.
    NB: l'agent utilise le `vectorstore` construit dans pipeline.py (via rag_search)."""
    rows = []
    for item in golden_set:
        t0 = time.time()
        answer = run_agent_fn(item["question"])
        elapsed = time.time() - t0
        if item.get("should_fail"):
            pertinente = is_refusal(item["question"], answer, llm)
        else:
            pertinente = score_answer(answer, item["expected_keywords"], item["question"])
        rows.append({
            "question": item["question"],
            "should_fail": bool(item.get("should_fail")),
            "latence_totale (s)": round(elapsed, 2),
            "reponse_pertinente": pertinente,
            "longueur_reponse": len(answer),
        })
    return pd.DataFrame(rows)
