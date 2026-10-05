# RAG sur la documentation Python 3.14

Agent RAG (Retrieval-Augmented Generation) qui répond à des questions sur Python en s'appuyant sur la documentation officielle, avec une étude comparative des stratégies de chunking et des modèles d'embedding.

## Architecture

```
config.py     configuration (variables d'environnement)
pipeline.py   chargement du corpus, chunking, embeddings, index Chroma, agent
eval.py       golden set, métriques de retrieval, grid search, évaluation bout en bout
RAG.ipynb   exécution et visualisation des résultats
```

- **Corpus** : fichiers `.txt` de la documentation Python (`CORPUS_DIR`).
- **Chunking** : `small_500` et `medium_1000` (RecursiveCharacterTextSplitter, en caractères), `token_based` (TokenTextSplitter, en tokens), `semantic` (SemanticChunker).
- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `BAAI/bge-small-en-v1.5`.
- **Base vectorielle** : Chroma, une collection par combinaison chunking × embedding, en distance cosinus.
- **Agent** : `ChatOllama` (`qwen2.5:7b`) avec un outil `rag_search`, qui reformule ses requêtes en anglais.
- **Interface** : chat Gradio au-dessus de l'agent (dernière section du notebook).

## Choix techniques

- **Une collection Chroma par combinaison testée** : sans nom de collection explicite, Chroma réutilise sa collection par défaut, partagée entre toutes les expériences. Chaque combo est ici indexé, évalué puis supprimé isolément.
- **Distance cosinus** (`collection_metadata={"hnsw:space": "cosine"}`) plutôt que la distance L2 par défaut de Chroma, pour que le score reflète la similarité sémantique.
- **Historique de conversation** passé à l'agent (`run_agent(..., history=...)`), pour l'interface de chat.

## Installation

```bash
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5:7b
ollama serve
```

## Utilisation

Lancer `RAG_clean.ipynb` de haut en bas :

1. Introduction & Objectifs
2. Préparation du Corpus
3. Expérimentations RAG (index, agent)
4. Évaluation & Résultats (grid search, graphiques, évaluation bout en bout)
5. Conclusion & Pistes d'amélioration
6. Interface de chat (Gradio)

## Résumé de l'étude comparative

Les métriques utilisées sont le Recall@3, le MRR, le score de similarité moyen, la latence de retrieval et le taux de troncature (part des chunks dépassant la fenêtre du modèle d'embedding).

## Limites connues

- Le jeu de test (`golden_set`, dans `eval.py`) contient des questions en français, alors que l'agent reformule ses requêtes en anglais avant de chercher dans une documentation anglaise.
- La pertinence d'un chunk est mesurée par présence de mots-clés (mots entiers), une approximation plus grossière qu'une correspondance sur la source attendue.
- Pas de recherche hybride (BM25 + vecteurs) ni de reranker.

