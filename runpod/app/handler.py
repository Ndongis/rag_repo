"""
handler.py
----------
Point d'entrée RunPod Serverless.
Au démarrage : clone le repo GitHub, charge les CSV, génère les embeddings.
À chaque requête : recherche RAG + réponse Gemini.

Variables d'environnement RunPod :
    GITHUB_RAG_REPO   https://github.com/user/rag-repo.git
    GEMINI_API_KEY    clé Gemini
    EMBED_MODEL       paraphrase-multilingual-mpnet-base-v2 (défaut)
"""

import csv
import hashlib
import json
import os
import subprocess

import google.generativeai as genai
import numpy as np
import runpod
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

GITHUB_RAG_REPO = os.getenv("GITHUB_RAG_REPO", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",  "")
EMBED_MODEL     = os.getenv("EMBED_MODEL",     "paraphrase-multilingual-mpnet-base-v2")

REPO_DIR   = "/tmp/rag-repo"
DATA_DIR   = f"{REPO_DIR}/data"
CACHE_EMB  = "/tmp/embeddings_cache.npy"
CACHE_META = "/tmp/metadata_cache.json"
CACHE_HASH = "/tmp/csv_hash.txt"

SYSTEM_PROMPT = """Tu es un guide expert et passionné de cette galerie virtuelle d'art.
Tu réponds en français, de façon naturelle, chaleureuse et engageante.
Tu t'appuies uniquement sur les œuvres et informations fournies dans le contexte.
Si une information n'est pas dans le contexte, dis-le honnêtement sans inventer.
Sois concis pour les réponses vocales (3-4 phrases maximum)."""

# ── État global (initialisé une fois au cold start) ───────────────────────────

_model:      SentenceTransformer = None
_llm:        genai.GenerativeModel = None
_embeddings: np.ndarray = None
_metadata:   list[dict] = []


# ── Git ───────────────────────────────────────────────────────────────────────

def clone_or_pull():
    if os.path.exists(f"{REPO_DIR}/.git"):
        print("Mise à jour du repo...")
        subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
    else:
        print(f"Clonage de {GITHUB_RAG_REPO}...")
        subprocess.run(["git", "clone", "--depth=1", GITHUB_RAG_REPO, REPO_DIR], check=True)


# ── CSV ───────────────────────────────────────────────────────────────────────

def read_csv(filename: str) -> list[dict]:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"WARN : {path} introuvable.")
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_index(rows: list[dict], key: str) -> dict:
    return {str(row.get(key, "")): row for row in rows}


def build_texts() -> tuple[list[dict], list[str]]:
    oeuvres     = read_csv("oeuvres.csv")
    artistes    = build_index(read_csv("artistes.csv"),    "user_id")
    salles      = build_index(read_csv("salles.csv"),      "id")
    expositions = build_index(read_csv("expositions.csv"), "id")

    print(f"  {len(oeuvres)} oeuvres | {len(artistes)} artistes | "
          f"{len(salles)} salles | {len(expositions)} expositions")

    metadata, texts = [], []

    for o in oeuvres:
        parts = [
            f"Titre : {o.get('titre', '')}",
            f"Auteur : {o.get('auteur', '')}",
            f"Date : {o.get('date', '')}",
            f"Technique : {o.get('technique', '')}",
            f"Sujet : {o.get('sujet', '')}",
        ]
        if o.get("inscription"):
            parts.append(f"Inscription : {o['inscription']}")
        if o.get("description_visuelle"):
            parts.append(f"Description visuelle : {o['description_visuelle']}")
        if o.get("historique"):
            parts.append(f"Historique : {o['historique']}")

        artiste = artistes.get(str(o.get("user_id", "")))
        if artiste:
            parts.append(
                f"Artiste : {artiste.get('prenom','')} {artiste.get('nom','')} "
                f"| Nationalité : {artiste.get('nationalite','')} "
                f"| Biographie : {artiste.get('biographie','')}"
            )

        salle = salles.get(str(o.get("salle_id", "")))
        if salle:
            parts.append(f"Salle : {salle.get('nom','')}")
            expo = expositions.get(str(salle.get("exposition_id", "")))
            if expo:
                parts.append(
                    f"Exposition : {expo.get('nom','')} — {expo.get('description','')}"
                )

        text = "\n".join(parts)
        texts.append(text)
        metadata.append({
            "oeuvre_id": o.get("id"),
            "titre":     o.get("titre", ""),
            "auteur":    o.get("auteur", ""),
            "contenu":   text,
        })

    return metadata, texts


# ── Cache embeddings ──────────────────────────────────────────────────────────

def csv_hash() -> str:
    h = hashlib.md5()
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".csv"):
            with open(os.path.join(DATA_DIR, f), "rb") as fp:
                h.update(fp.read())
    return h.hexdigest()


def load_or_build() -> tuple[np.ndarray, list[dict]]:
    current_hash = csv_hash()

    if (os.path.exists(CACHE_EMB) and
            os.path.exists(CACHE_META) and
            os.path.exists(CACHE_HASH)):
        if open(CACHE_HASH).read().strip() == current_hash:
            print("Cache valide — embeddings chargés.")
            return np.load(CACHE_EMB), json.loads(open(CACHE_META).read())

    print("Génération des embeddings...")
    metadata, texts = build_texts()
    embeddings = _model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    np.save(CACHE_EMB, embeddings)
    open(CACHE_META, "w").write(json.dumps(metadata))
    open(CACHE_HASH, "w").write(current_hash)

    print(f"Embeddings prêts ({len(metadata)} oeuvres).")
    return embeddings, metadata


# ── Recherche cosinus ─────────────────────────────────────────────────────────

def cosine_search(question: str, top_k: int = 4) -> list[dict]:
    q      = _model.encode(question)
    q      = q / (np.linalg.norm(q) + 1e-10)
    norms  = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-10
    scores = (_embeddings / norms) @ q
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{**_metadata[i], "score": float(scores[i])} for i in top_idx]


# ── Génération Gemini ─────────────────────────────────────────────────────────

def generate_answer(question: str, results: list[dict]) -> str:
    if not results or not _llm:
        return "Je n'ai pas trouvé d'œuvre correspondant à votre question."

    context = "\n\n---\n\n".join(
        f"Œuvre : {r['titre']} | Auteur : {r['auteur']}\n{r['contenu']}"
        for r in results
    )
    prompt = f"{SYSTEM_PROMPT}\n\nContexte :\n{context}\n\nQuestion : {question}"
    return _llm.generate_content(prompt).text


# ── Initialisation cold start ─────────────────────────────────────────────────

def initialize():
    global _model, _llm, _embeddings, _metadata

    print("=== STEP 1 : clone repo ===")
    if GITHUB_RAG_REPO:
        clone_or_pull()
    else:
        print("WARN : GITHUB_RAG_REPO non défini.")

    print("=== STEP 2 : load model ===")
    _model = SentenceTransformer(EMBED_MODEL)

    print("=== STEP 3 : build embeddings ===")
    _embeddings, _metadata = load_or_build()

    print("=== STEP 4 : configure Gemini ===")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _llm = genai.GenerativeModel("gemini-1.5-flash")
        print("Gemini configuré.")
    else:
        print("WARN : GEMINI_API_KEY non définie.")

    print(f"=== READY : {len(_metadata)} oeuvres indexées ===")

# ── Handler RunPod ────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """
    Point d'entrée RunPod Serverless.

    Format d'entrée attendu (depuis VAPI tool call) :
    {
        "input": {
            "question": "Qui a peint La Joconde ?",
            "top_k": 4   (optionnel)
        }
    }

    Format de sortie :
    {
        "answer": "...",
        "sources": [{"titre": "...", "auteur": "...", "score": 0.92}]
    }
    """
    job_input = job.get("input", {})
    question  = job_input.get("question", "")
    top_k     = int(job_input.get("top_k", 4))

    if not question:
        return {"error": "Paramètre 'question' manquant."}

    results = cosine_search(question, top_k=top_k)
    answer  = generate_answer(question, results)

    return {
        "answer": answer,
        "sources": [
            {"titre": r["titre"], "auteur": r["auteur"], "score": r["score"]}
            for r in results
        ],
    }


# ── Démarrage ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initialize()
    runpod.serverless.start({"handler": handler})