# Audit technique – AISCA (Streamlit RAG local)

## Portée
- Code base analysée : `app.py`, `src/` (notamment `scoring_v2.py`, `genai_client.py`, `semantic_engine.py`, `data_loader.py`, `cache.py`), données `data/*.json`, dépendances `requirements.txt`, pages Streamlit (structure observée). Aucun changement de code effectué.
- Objectif : évaluer architecture, données, pipeline de scoring v2, dépendances, risques, et proposer des recommandations actionnables.

## Architecture et flux
- **UI** : app Streamlit multipage (`app.py` + `pages/`). Page d’accueil qui route vers `pages/1_Introduction.py` via bouton.
- **Données référentiel v1 (sémantique)** : `data/competency_blocks.json` (6 blocs pondérés) + `data/job_profiles.json` (8 métiers). Utilisés par `SemanticEngine` pour embeddings SBERT (`sentence-transformers/all-MiniLM-L6-v2`) et matching blocs/métiers (scores cosinus).
- **Données référentiel v2 (règles déterministes)** : `data/skills_v2.json` (compétences avec `block_id`) + `data/jobs_v2.json` (5 métiers, required/optional skills). Utilisés par `src/utils/scoring_v2.py`.
- **LLM** : appels locaux via Ollama (`phi3:mini` par défaut) avec fallback `llama.cpp` CLI. Prompts contrôlés dans `genai_client.py` (bio, plan, enrichissement). Caching des réponses dans `.cache/genai_outputs`.
- **Caching embeddings** : `SemanticEngine` met en cache embeddings (diskcache) dans `.cache/embeddings` avec clé hashée (texte + modèle) pour accélérer relances.
- **Persistences** : exports des réponses dans `outputs/*.json|csv`.

## Pipeline scoring_v2 (règles déterministes)
- **Entrée** : `session_state.responses` ou dict déjà structuré. Clés utilisées :
  - `likert` (valeurs 1–5) mappées via `SKILL_LIKERT_MAP` vers des compétences (ex: `python_level` → `python`, `etl_level` → `etl_elt`/`data_modeling`/... etc.).
  - `selected_skills` (checkbox) et `qcm_multi` (outils) → sélection explicite.
  - Textes libres : `free_text`, `ambitions`, `proj_tech`, `prob_complex`, `model_build`, `projection`.
- **Score compétence** (`compute_skill_scores`) :
  - Auto-évaluation : note Likert clampée 1–5 → linéaire 0..70.
  - Sélection explicite : +25 si mot-clé du skill présent dans `selected_skills` ou `tools`.
  - Texte libre : +15 si mots-clés (_skill_id_, _skill_name_ ou overrides) matchent le corpus normalisé.
  - Score borné à 100. Tri décroissant. Sources conservées.
- **Score métier** (`compute_job_scores`) : moyenne pondérée 75% `required` + 25% `optional` (moyennes simples des scores de compétences). Tri décroissant.
- **Coverage** (`compute_coverage`) : % de compétences requises avec score ≥ 60 (par défaut).
- **Niveau** (`classify_level`) : <45 Junior, ≤70 Mid, sinon Senior.
- **Génération texte** (`_generate_texts`) : bio + explication via LLM si disponible, sinon fallback statique. Contrôles : catch `GenAIUnavailable`.
- **Résumé** : `run_analysis_v2` assemble `meta` (version, durée, signature SHA256 des inputs), `inputs_resume` (compte champs), `job_scores`, `skill_scores`, `coverage_score`, `summary_profile`, `professional_bio`, `debug` (top jobs/skills, inputs, signature).

## Points forts
- 100% local (LLM + embeddings) : pas de dépendance API externe ni coût variable.
- Caching généralisé (diskcache) pour LLM et embeddings → relances rapides et idempotentes.
- Scoring v2 lisible, déterministe et borné ; sources de contribution conservées pour auditabilité.
- Prompts LLM explicites avec instructions anti-hallucination et séparation claire des tâches (bio, plan, enrichissement).
- Données référentiel structurées et faciles à étendre (JSON). Versionnement implicite entre v1 (blocs) et v2 (compétences).

## Risques / faiblesses identifiés
1) **Robustesse des entrées** : aucune validation forte des `session_state.responses` (types, bornes, présence de clés). Des valeurs non numériques ou manquantes tombent à 0 silencieusement.
2) **Matching texte par mots-clés simples** : `_match_any` sur texte normalisé → faux positifs/negatifs possibles ; pas de stemming ni seuil de similarité.
3) **Poids fixes** : +25 sélection / +15 texte / 0..70 auto-éval → arbitrage non testé ; pas de calibration empirique ni tests unitaires pour justifier seuils (coverage 60, niveau 45/70).
4) **Couverture** : jobs sans `required_skills` retourneraient 0 (safe) mais non signalé ; aucune vérification que toutes les compétences référencées existent dans `skills_v2.json`.
5) **LLM disponibilité** : si Ollama down et `llama-cli` absent, exceptions remontent (log) mais pas d’affichage utilisateur clair hors fallback du texte ; risque d’expérience dégradée.
6) **Sécurité / confidentialité** :
   - Pas d’authentification sur l’app Streamlit.
   - Données utilisateurs (réponses) écrites en clair dans `outputs/` sans rotation/PII scrubbing.
   - Prompts peuvent contenir données sensibles ; logs potentiels via Ollama/CLI.
7) **Qualité logicielle** :
   - Peu de tests automatisés visibles (dossier `tests/` non renseigné ici).
   - Pas de lint/CI décrit ; pas de contrôle de version des modèles (SBERT) ni vérification des fichiers data v2 vs v1.
8) **Perf / UX** :
   - Chargement modèle SBERT au démarrage (peut prendre quelques secondes) ; pas de préchauffage conditionnel.
   - Pas de monitoring de durée des appels LLM côté UX (seulement logs). 

## Recommandations prioritaires (rapides)
1) **Validation des entrées** (haute priorité)
   - Schéma (pydantic ou validation custom) pour `responses` : types, bornes 1–5, listes de strings non vides.
   - Vérifier que les `skill_id` référencées par jobs existent dans `skills_v2.json` au load, sinon log/raise explicite.

2) **Observabilité et UX LLM**
   - Surface côté UI l’état LLM : disponible / fallback / indisponible, avec message utilisateur clair.
   - Loguer durée et succès/échec des appels LLM dans `run_analysis_v2` (meta) et afficher durée en debug UI.

3) **Sécurité / données**
   - Option pour désactiver l’écriture `outputs/` ou anonymiser (hash email, retirer PII texte) ; prévoir rotation/retention.
   - Ajouter avertissement de confidentialité sur la page d’accueil.

4) **Amélioration scoring v2**
   - Paramétrer les poids (auto, sélection, texte) et le seuil coverage via config (ex: `settings.yaml`).
   - Introduire une étape de similarité légère (embeddings minilm) pour le texte libre pour réduire faux positifs de mots-clés.

5) **Qualité / tests**
   - Ajouter tests unitaires couvrant : `_auto_eval_score`, `_match_any`, `compute_skill_scores` (cas selection, texte, likert), `compute_job_scores` (poids required/optional) et `compute_coverage` (seuils). 
   - Ajouter un test d’intégration `self_check` automatisé en CI pour valider top métiers attendus.

6) **Opérations**
   - Script de préflight (check Ollama dispo, modèle présent, accès disque cache) et indicateur dans UI.
   - Fichier `Makefile` ou task runner (`just`) pour setup/lancement/test.

## Pistes moyen terme
- Normaliser les données v2 : sérialiser schéma JSON (ex: `pydantic.BaseModel`) pour compétences/métiers afin de détecter ruptures de contrat.
- Harmoniser v1/v2 : choisir entre pipeline sémantique et scoring déterministe, ou les combiner (hybride) avec ablation tests pour valider gains.
- Exposer un endpoint interne (FastAPI) pour scoring afin de séparer logique métier de la couche UI Streamlit.
- Ajouter instrumentation légère (OpenTelemetry/structured logging) pour mesurer temps modèle, cache hit rate, et distribution des scores.

## Checklist de conformité rapide
- **Dépendances** : `requirements.txt` minimal, pas de GPU requis (torch CPU ok). ✔️
- **Données locales uniquement** : pas d’appel externe par défaut (Ollama local). ✔️
- **Gestion des secrets** : aucune clé API requise. ✔️
- **Tests** : non présents/visibles. ❌
- **Auth** : aucune, app publique sur le port si exposée. ❌

## Résumé exécutif
AISCA est une app Streamlit locale qui combine un pipeline sémantique (SBERT + référentiel blocs v1) et un scoring déterministe v2 (JSON compétences/métiers) enrichi par un LLM local pour le storytelling (bio/plan). Les fondations sont claires et locales, mais la robustesse d’entrée, la calibration des scores, la gestion de la confidentialité et l’absence de tests/CI sont les principaux risques. Les actions rapides listées ci-dessus renforceront la fiabilité perçue et la sécurité sans complexifier l’architecture.
