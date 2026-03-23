#!/bin/bash
# startup.sh — RunPod
# Variables requises : GITHUB_RAG_REPO, GEMINI_API_KEY

set -e
REPO_DIR="/workspace/repo"

echo "========================================"
echo " RAG Service — Galerie Virtuelle"
echo "========================================"

# 1. Clone ou pull (repo public — pas de token)
if [ -d "$REPO_DIR/.git" ]; then
    echo "[1/3] Mise à jour du repo..."
    cd "$REPO_DIR" && git pull
else
    echo "[1/3] Clonage du repo public..."
    git clone "$GITHUB_RAG_REPO" "$REPO_DIR"
fi

# Vérifier les CSV
if [ ! -f "$REPO_DIR/data/oeuvres.csv" ]; then
    echo "ERREUR : data/oeuvres.csv introuvable."
    echo "Lance d'abord POST /export sur l'export-service."
    exit 1
fi

echo "CSV disponibles :"
ls -lh "$REPO_DIR/data/"*.csv

# 2. Installer les dépendances
echo "[2/3] Installation des dépendances..."
pip install --quiet -r "$REPO_DIR/runpod/requirements.txt"

# 3. Lancer le RAG
echo "[3/3] Démarrage du RAG sur le port 8000..."
export DATA_DIR="$REPO_DIR/data"
export GEMINI_API_KEY="$GEMINI_API_KEY"

cd "$REPO_DIR/runpod/app"
uvicorn runpod_rag:app --host 0.0.0.0 --port 8000