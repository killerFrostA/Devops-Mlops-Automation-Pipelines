# Agent 1 – Résultats des tests

Date du test réel : 11/10/2026, 19:05 à 19:44, sur le banc local (service de paiement de test, Prometheus, cAdvisor).
Script : `python experiments/checks/full_scenario.py 19:05`. Les journaux complets sont dans `experiments/runs/scenario/` (non versionnés).

## 1. Tests automatiques

| Contrôle | Résultat |
|---|---|
| `pytest tests` (sans Docker) | 227 réussis, 1 ignoré par défaut |
| Test avec le vrai modèle Chronos (`PIDS_RUN_CHRONOS=1`) | réussi : courbe stable jugée normale, fuite jugée anormale |
| ruff, mypy (strict), `export_contracts --check` | aucune erreur |

Chaque événement produit par l'agent est validé par le contrat Python **et** par le schéma JSON `contracts/jsonschema/v1/EventEnvelope.schema.json`.

## 2. Délai de détection par panne (test réel)

Pour chaque panne, on lance un nouvel agent au moment de l'injection, avec un cycle toutes les 30 s.

| Panne injectée | Détecteur | Première détection | Délai |
|---|---|---|---|
| Fuite mémoire 30 Mo/min | Statistique (pente) | 19:07:50 | 2 min 50 s |
| | Chronos | 19:08:20 | 3 min 20 s |
| Latence +800 ms | SLO latence (slow-burn) | 19:19:17 | 2 min 23 s |
| | Statistique (saut de niveau) | 19:19:47 | 2 min 53 s |
| | SLO latence (fast-burn, HIGH) | 19:20:17 | 3 min 23 s |
| | Chronos | 19:20:17 | 3 min 23 s |
| 50 % d'erreurs HTTP 500 | SLO disponibilité (slow-burn) | 19:30:41 | 5 min 21 s |
| | SLO disponibilité (fast-burn, HIGH) | 19:31:11 | 5 min 51 s |
| CPU saturé | Statistique (saut de niveau) | 19:38:35 | 2 min 52 s |
| | Chronos | 19:39:05 | 3 min 22 s |

**Lecture :**
- Les 4 pannes sont détectées en **2 min 23 s à 5 min 51 s**.
- La statistique et Chronos détectent toutes les pannes de mémoire, de latence et de CPU. Chronos arrive en moyenne **30 s après** la statistique.
- Les erreurs 500 sont vues par le SLO uniquement : c'est voulu, car un taux d'erreurs se juge par rapport à l'objectif de qualité.
- Le délai du SLO sur les erreurs est plus long (environ 5 min 30 s). C'est le prix des fenêtres longues (1 h et 6 h), qui évitent d'alerter sur un pic bref.

## 3. Fausses alertes

- **Aucune alerte sur un signal sain** pendant une panne d'un autre type : par exemple, le CPU et la latence restent normaux pendant la fuite mémoire.
- **Signaux résiduels :** au début des phases C et D, l'agent signale encore la panne de la phase précédente (latence, puis erreurs) pendant 20 à 30 s. Ce ne sont pas des fausses alertes : les fenêtres SLO de 5 minutes contiennent encore les requêtes dégradées. La fenêtre courte de 5 minutes garantit que ces signaux s'arrêtent après la guérison.

## 4. Envoi aux autres agents (contrat v1)

9 événements ont été écrits dans `events.jsonl`, tous valides :

| Sujet | Nombre | Exemple |
|---|---|---|
| `monitoring.anomaly.detected` | 7 | fuite mémoire : P2 / MEDIUM, confiance 0,84, détecteur `statistical` |
| `monitoring.recovery.verified` | 2 | voir section 5 |

La priorité suit la gravité : les incidents HIGH (SLO fast-burn) partent en P1. Le même incident garde la même clé d'idempotence.

## 5. Vérification indépendante de la guérison

Le message `deployment.action.completed` est fourni par le fichier `experiments/checks/action_completed.json`. Fenêtre d'observation : 2 min, avec une mesure toutes les 30 s.

| Situation | Résultat de l'agent |
|---|---|
| L'action se dit réussie mais la fuite continue | **NOT_RECOVERED** (`memory_slope_mb_min` = FAIL) |
| Après la vraie réparation | **RECOVERED** (erreurs, latence et mémoire = PASS) |

L'agent ne croit pas l'agent de déploiement sur parole : il mesure lui-même.

## 6. Tâche de l'Orchestrateur

`inbox detect payment-service`, envoyée pendant la fuite, a renvoyé un `AgentResult` SUCCEEDED : « Anomalie sur payment-service : memory_pressure », avec une confiance de 0,6 et une preuve `ANM-…`.

## 7. Limites connues

- **Chronos s'habitue à une panne longue.** Après environ 6 minutes de fuite, la fuite entre dans son historique et il la prédit. La statistique (pente) prend le relais. Amélioration possible : figer l'historique de référence pendant un incident.
- **Kafka et le Context Service de l'équipe ne sont pas encore implémentés.** Les événements passent pour l'instant par un fichier JSON Lines, à travers l'interface `EventPublisher` : le branchement sur Kafka ne changera pas l'agent.
- **Kubernetes et les traces ne sont pas couverts.** Le banc local surveille uniquement des conteneurs Docker.
