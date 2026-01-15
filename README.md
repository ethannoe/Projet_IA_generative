# AISCA – Mini-agent RAG de cartographie de compétences

Application Streamlit locale qui :
- Collecte un questionnaire hybride (Likert, QCM, cases à cocher, texte libre).
- Calcule des similarités sémantiques via SBERT local (`all-MiniLM-L6-v2`).
- Fait du matching blocs/compétences et métiers (coverage score + top 3 métiers).
- Guide un LLM local (Ollama, modèle léger `phi3:mini` par défaut) avec un contexte RAG.
- Génère un plan de progression et une bio professionnelle en 1 appel chacun (cachés).

## Arborescence
```
app.py
src/
  cache.py
  data_loader.py
  semantic_engine.py
  scoring.py
  rag_agent.py
  genai_client.py
  viz.py
data/
  competency_blocks.json
  job_profiles.json
outputs/                # réponses utilisateur et exports CSV
requirements.txt
```

## Installation rapide (Mac, venv conseillé)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancement
```bash
streamlit run app.py
```

## Modèle génératif local (gratuit, sans clé)
- **Ollama (préféré)** : installer depuis <https://ollama.com>. Puis :
```bash
ollama pull phi3:mini
```
- L’app vérifie la disponibilité d’Ollama. Si absent, un message s’affiche. Un fallback minimal via `llama.cpp` (CLI `llama-cli`) est prévu mais non requis.

## Données de référentiel
- `data/competency_blocks.json` : 6 blocs (DA, ML, NLP, DE, PM, MLOps) avec poids et phrases courtes de compétences.
- `data/job_profiles.json` : 8 métiers avec blocs requis + poids.
- Extensibilité : ajoutez des blocs/compétences ou métiers en conservant la structure JSON.

## Pipeline sémantique
- **Embedding** : `sentence-transformers/all-MiniLM-L6-v2` chargé localement, cache `.models/`.
- **Similarité** : cosinus entre textes utilisateur et compétences du référentiel.
- **Score bloc (simple)** : pour chaque texte utilisateur, on prend la meilleure similarité avec les compétences du bloc ; le score du bloc est la moyenne de ces meilleures similarités.
- **Coverage Score** : $\text{Coverage} = \frac{\sum W_i \times S_i}{\sum W_i}$, borné [0,1].
- **Interprétation** : `>=0.7` forte, `0.5-0.7` moyenne, `<0.5` faible.
- **Matching métiers** : moyenne pondérée des blocs requis (Job Fit = Σ w·S / Σ w). Justification simple listant les blocs dominants et les contributions.

## Mini-agent RAG
- Récupère les top compétences proches (top-k) et construit un *context pack* injecté dans les prompts LLM.
- Le LLM est explicitement bridé : "Utilise uniquement le référentiel / si info manquante, dis-le".

## Génération contrôlée (LLM)
- Aucune clé API requise, appels locaux uniquement.
- **Limites** :
  - Enrichissement optionnel si texte < 5 mots (1 appel max, activable/désactivable via UI).
  - Plan de progression : 1 bouton → 1 appel → résultat mis en cache.
  - Bio : 1 bouton → 1 appel → résultat mis en cache.
- **Caching GenAI** : hash du prompt + modèle + température stocké dans `.cache/genai_outputs` (diskcache). Répéter la même requête renvoie le résultat existant.

## Caching embeddings
- Embeddings des compétences et des textes utilisateur sont stockés dans `.cache/embeddings` avec clé hashée (texte + modèle). Relance rapide et économe en RAM.

## Visualisations
- Radar : scores par bloc (0-1).
- Bar chart : top métiers (score de matching 0-1).

## Fichiers de sortie
- `outputs/user_responses.json` + `outputs/user_responses.csv` : réponses brutes du questionnaire.

## Profils de démonstration (pré-remplissage suggéré)

### Use Case 1 – Data Scientist Confirmé
- Python, pandas, numpy, dashboards, régression & classification, NLP basique (tokenization, embeddings)
- Réponses clés : Likert élevés sur ML/NLP/Pipelines, guidées « oui » pour prod/app & bibliothèques NLP, texte libre mentionnant modèles + dashboards.
- Attendu : Coverage ≈ 0.75–0.85 ; métiers : Data Scientist, ML Engineer, NLP Engineer.

### Use Case 2 – ML Engineer / MLOps Junior
- Modélisation ML, pipelines, déploiement, peu de NLP
- Réponses clés : Likert fort en pipelines/ML/MLops, guidées « oui » sur déploiement et big data, NLP modéré.
- Attendu : Coverage ≈ 0.65–0.75 ; métiers : ML Engineer, Data Engineer, Data Scientist.

### Use Case 3 – Analyste Junior / Reconversion
- SQL, Excel, Python basique, visualisation, très peu de ML
- Réponses clés : Likert moyens/faibles en ML, forts en data/communication, guidées surtout « non » sur prod/LLM, texte libre orienté reporting.
- Attendu : Coverage ≈ 0.45–0.60 ; métiers : Data Analyst, BI Analyst, Junior Data Scientist.

## Prompts (résumés)
- **Enrichissement** (optionnel) : réécrire une phrase trop courte sans inventer, ou indiquer manque d’info.
- **Plan de progression** : étapes 4-8 semaines, basé sur compétences manquantes, référentiel uniquement, bullets + jalons.
- **Bio** : Executive Summary 4-6 lignes, mention blocs forts + métiers compatibles, aucune invention.

## Notes performance (Mac Air 8Go)
- Modèle SBERT léger, embeddings mis en cache.
- Modèle LLM léger (`phi3:mini` ou équivalent <=4B) conseillé.
- Pas de clé, tout en local.

## Dépannage
- Ollama non détecté : installez Ollama et exécutez `ollama pull phi3:mini`, puis relancez l'app.
- Import errors : assurez-vous d’activer le venv et d’installer les requirements.
