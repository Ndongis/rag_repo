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
Tu t'appuies uniquement sur les informations fournies dans le contexte.
Si une information n'est pas dans le contexte, dis-le honnêtement sans inventer.

Quand tu parles d'une oeuvre : décris ses caractéristiques, parle de l'artiste et du contexte.
Quand tu parles d'un artiste : parle de son parcours, sa nationalité, ses oeuvres dans la galerie.
Quand tu parles d'une exposition : décris-la, cite les salles et les oeuvres présentées.
Sois concis pour les réponses vocales (3-5 phrases maximum)."""

_model:      SentenceTransformer = None
_llm:        genai.GenerativeModel = None
_embeddings: np.ndarray = None
_metadata:   list[dict] = []


def clone_or_pull():
    if os.path.exists(f"{REPO_DIR}/.git"):
        print("Mise à jour du repo...")
        subprocess.run(["git", "-C", REPO_DIR, "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "origin/main"], check=True)
    else:
        print(f"Clonage de {GITHUB_RAG_REPO}...")
        subprocess.run(["git", "clone", "--depth=1", GITHUB_RAG_REPO, REPO_DIR], check=True)


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
    artistes    = read_csv("artistes.csv")
    salles      = read_csv("salles.csv")
    expositions = read_csv("expositions.csv")

    artistes_idx    = build_index(artistes,    "user_id")
    salles_idx      = build_index(salles,      "id")
    expositions_idx = build_index(expositions, "id")

    print(f"  {len(oeuvres)} oeuvres | {len(artistes)} artistes | "
          f"{len(salles)} salles | {len(expositions)} expositions")

    metadata, texts = [], []

    # Documents Oeuvres
    for o in oeuvres:
        parts = [
            "[OEUVRE]",
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

        artiste = artistes_idx.get(str(o.get("user_id", "")))
        if artiste:
            parts.append(
                f"Artiste : {artiste.get('prenom','')} {artiste.get('nom','')} "
                f"| Nationalité : {artiste.get('nationalite','')} "
                f"| Biographie : {artiste.get('biographie','')}"
            )

        salle = salles_idx.get(str(o.get("salle_id", "")))
        if salle:
            parts.append(f"Salle : {salle.get('nom','')}")
            expo = expositions_idx.get(str(salle.get("exposition_id", "")))
            if expo:
                parts.append(
                    f"Exposition : {expo.get('nom','')} — {expo.get('description','')}"
                )

        text = "\n".join(parts)
        texts.append(text)
        metadata.append({"type": "oeuvre", "titre": o.get("titre", ""), "auteur": o.get("auteur", ""), "contenu": text})

    # Documents Artistes
    for a in artistes:
        oeuvres_artiste = [
            o.get("titre", "") for o in oeuvres
            if str(o.get("user_id", "")) == str(a.get("user_id", ""))
        ]
        parts = [
            "[ARTISTE]",
            f"Nom : {a.get('prenom','')} {a.get('nom','')}",
            f"Nationalité : {a.get('nationalite','')}",
            f"Date de naissance : {a.get('date_naissance','')}",
            f"Biographie : {a.get('biographie','')}",
        ]
        if oeuvres_artiste:
            parts.append(f"Oeuvres dans la galerie : {', '.join(oeuvres_artiste)}")
        text = "\n".join(parts)
        texts.append(text)
        metadata.append({"type": "artiste", "titre": f"{a.get('prenom','')} {a.get('nom','')}", "auteur": "", "contenu": text})

    # Documents Expositions
    for e in expositions:
        salles_expo = [s.get("nom", "") for s in salles if str(s.get("exposition_id", "")) == str(e.get("id", ""))]
        salles_ids  = [str(s.get("id", "")) for s in salles if str(s.get("exposition_id", "")) == str(e.get("id", ""))]
        oeuvres_expo = [o.get("titre", "") for o in oeuvres if str(o.get("salle_id", "")) in salles_ids]
        parts = [
            "[EXPOSITION]",
            f"Nom : {e.get('nom','')}",
            f"Description : {e.get('description','')}",
        ]
        if salles_expo:
            parts.append(f"Salles : {', '.join(salles_expo)}")
        if oeuvres_expo:
            parts.append(f"Oeuvres présentées : {', '.join(oeuvres_expo)}")
        text = "\n".join(parts)
        texts.append(text)
        metadata.append({"type": "exposition", "titre": e.get("nom", ""), "auteur": "", "contenu": text})

    # Documents Salles
    for s in salles:
        oeuvres_salle = [o.get("titre", "") for o in oeuvres if str(o.get("salle_id", "")) == str(s.get("id", ""))]
        expo = expositions_idx.get(str(s.get("exposition_id", "")))
        parts = ["[SALLE]", f"Nom : {s.get('nom','')}"]
        if expo:
            parts.append(f"Exposition : {expo.get('nom','')} — {expo.get('description','')}")
        if oeuvres_salle:
            parts.append(f"Oeuvres dans cette salle : {', '.join(oeuvres_salle)}")
        text = "\n".join(parts)
        texts.append(text)
        metadata.append({"type": "salle", "titre": s.get("nom", ""), "auteur": "", "contenu": text})

    print(f"  Total documents RAG : {len(texts)}")
    return metadata, texts


def csv_hash() -> str:
    h = hashlib.md5()
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".csv"):
            with open(os.path.join(DATA_DIR, f), "rb") as fp:
                h.update(fp.read())
    return h.hexdigest()


def load_or_build() -> tuple[np.ndarray, list[dict]]:
    current_hash = csv_hash()
    if (os.path.exists(CACHE_EMB) and os.path.exists(CACHE_META) and os.path.exists(CACHE_HASH)):
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
    print(f"Embeddings prêts ({len(metadata)} documents).")
    return embeddings, metadata


def cosine_search(question: str, top_k: int = 4) -> list[dict]:
    q      = _model.encode(question)
    q      = q / (np.linalg.norm(q) + 1e-10)
    norms  = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-10
    scores = (_embeddings / norms) @ q
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{**_metadata[i], "score": float(scores[i])} for i in top_idx]


def generate_answer(question: str, results: list[dict]) -> str:
    if not results or not _llm:
        return "Je n'ai pas trouvé d'information correspondant à votre question."
    context = "\n\n---\n\n".join(
        f"[{r.get('type','').upper()}] {r['titre']}\n{r['contenu']}"
        for r in results
    )
    prompt = f"{SYSTEM_PROMPT}\n\nContexte :\n{context}\n\nQuestion : {question}"
    return _llm.generate_content(prompt).text


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
        _llm = genai.GenerativeModel("gemini-2.5-flash")
        print("Gemini configuré.")
    else:
        print("WARN : GEMINI_API_KEY non définie.")

    print(f"=== READY : {len(_metadata)} documents indexés ===")


def extract_question(job_input: dict) -> tuple:
    message    = job_input.get("message", {})
    tool_calls = message.get("toolCalls", [])
    if tool_calls:
        tool_call    = tool_calls[0]
        tool_call_id = tool_call.get("id")
        args         = tool_call.get("function", {}).get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return args.get("question", ""), tool_call_id
    return job_input.get("question", ""), None


def handler(job: dict) -> dict:
    job_input              = job.get("input", {})
    question, tool_call_id = extract_question(job_input)
    top_k                  = int(job_input.get("top_k", 4))

    if not question:
        return {"error": "Paramètre 'question' manquant."}

    results = cosine_search(question, top_k=top_k)
    answer  = generate_answer(question, results)

    if tool_call_id:
        return {"results": [{"toolCallId": tool_call_id, "result": answer}]}

    return {
        "answer": answer,
        "sources": [{"type": r.get("type",""), "titre": r["titre"], "score": r["score"]} for r in results],
    }

"""
if __name__ == "__main__":
    initialize()
    runpod.serverless.start({"handler": handler})
"""

# Remplacer le démarrage RunPod par FastAPI pour un pod
import json
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
api = FastAPI()

class QueryRequest(BaseModel):
    question: str
    oeuvre_id: str = ""
    top_k: int = 4

@api.get("/health")
def health():
    return {"status": "ok", "documents": len(_metadata)}

@api.post("/query")
def query(req: QueryRequest):
    results = cosine_search(req.question, req.top_k)
    answer  = generate_answer(req.question, results)
    return {"answer": answer, "sources": [{"titre": r["titre"], "score": r["score"]} for r in results]}

@api.post("/vapi/webhook")
async def vapi_webhook(request: Request):
    body       = await request.json()
    tool_calls = body.get("message", {}).get("toolCalls", [])
    if not tool_calls:
        return {"error": "Aucun tool call"}
    tool_call    = tool_calls[0]
    tool_call_id = tool_call.get("id")
    args         = tool_call.get("function", {}).get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    question = args.get("question", "")
    results  = cosine_search(question, top_k=4)
    answer   = generate_answer(question, results)
    return {"results": [{"toolCallId": tool_call_id, "result": answer}]}

# Remplacer le if __name__ == "__main__" par :
if __name__ == "__main__":
    import uvicorn
    initialize()
    uvicorn.run(api, host="0.0.0.0", port=8000)