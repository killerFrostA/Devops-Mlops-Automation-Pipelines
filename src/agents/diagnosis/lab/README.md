# Labo de l’agent 2 : vraies pannes, vrais outils

Un petit cluster Kubernetes local (`kind`) dans lequel on provoque de **vraies** pannes, puis
l’agent 2 les diagnostique avec ses outils en direct (Prometheus, Loki, `kubectl`, `git`).
Tout est dans ce dossier ; aucun fichier commun du projet n’est modifié.

## Ce qui tourne

| Namespace | Composant | Rôle |
|---|---|---|
| `prod` | `payment-service` | application de test (Python) ; ses pannes viennent de sa configuration |
| `prod` | `ledger-db` | dépendance TCP (port 5432) de payment-service |
| `prod` | `loadgen` | trafic continu (~6 requêtes/s) |
| `monitoring` | Prometheus | métriques cAdvisor + kube-state-metrics + `/metrics` de l’app — `http://127.0.0.1:9091` |
| `monitoring` | kube-state-metrics | limites mémoire et redémarrages des pods |
| `monitoring` | Loki + Fluent Bit | logs des conteneurs `prod` — `http://127.0.0.1:3100` |

Le dépôt git `out/payment-service-repo` joue le rôle du dépôt de l’application : chaque
« release » fautive y est commitée.

## Budget mémoire (PC 8 Go, WSL limité à 4 Go)

Un seul nœud kind, une seule panne à la fois (`inject` refuse tant que `reset` n’a pas été fait).

| Composant | Limite |
|---|---:|
| Nœud kind (API server, etcd, kubelet…) | ~0,8–1,2 Go, non plafonné |
| Prometheus (collecte 30 s, 1 jour d’historique) | 256 Mi |
| Loki | 192 Mi |
| kube-state-metrics, Fluent Bit | 64 Mi chacun |
| payment-service (la fuite le fait tuer à cette limite) | 96 Mi |
| ledger-db, loadgen | 48 Mi chacun |
| Qdrant du compose (`lab qdrant start`, plafond posé avec `docker update`) | 256 Mo |

Total au pire ≈ 2,3 Go sur les ~3,8 Gio de Docker. `lab qdrant stop` libère Qdrant quand le RAG
n’est pas utilisé ; `lab down` supprime tout le cluster.

## Pannes disponibles

| Panne | Ce qui se passe vraiment | Cause attendue | 1re action attendue |
|---|---|---|---|
| `leak` | release 1.4.0 qui garde chaque réponse en mémoire → croissance puis OOMKilled | `MEMORY_LEAK_POST_DEPLOY` | `ROLLBACK_OR_REVERT_DEPLOYMENT` |
| `bad-config` | release 1.4.1 avec `PAYMENT_TIMEOUT_MS=5s` → le nouveau pod quitte au démarrage (CrashLoopBackOff) | `CONFIG_ERROR_CRASHLOOP` | `ROLLBACK_OR_REVERT_DEPLOYMENT` |
| `dependency-down` | `ledger-db` mis à 0 réplique → « connection refused », sans aucune release | `DEPENDENCY_UNAVAILABLE` | `INVESTIGATE_DEPENDENCY` |
| `pool` | release 1.4.2 avec un pool de 1 connexion et des requêtes lentes → pool épuisé | `CONNECTION_POOL_EXHAUSTION` | `TUNE_CONNECTION_POOL` |

## Utilisation (depuis la racine du dépôt)

```powershell
py -3 -m src.agents.diagnosis.lab.lab check            # outils + mémoire Docker
py -3 -m src.agents.diagnosis.lab.lab up               # crée le cluster et déploie (≈ 5–10 min la 1re fois)
py -3 -m src.agents.diagnosis.lab.lab status
py -3 -m src.agents.diagnosis.lab.lab inject leak
py -3 -m src.agents.diagnosis.lab.lab run leak --wait --qdrant   # attend l’anomalie, puis diagnostic
py -3 -m src.agents.diagnosis.lab.lab reset
py -3 -m src.agents.diagnosis.lab.lab down             # supprime le cluster
```

`run` fait deux choses :

1. **Il joue l’agent 1** : il lit les signaux dans Prometheus (mémoire, pente, latence, taux
   d’erreur, redémarrages) et écrit un événement `ANOMALY_DETECTED` dans `out/incidents/`, avec un
   fichier `.truth.json` (la vraie cause, pour l’évaluation). Par défaut, l’événement **ne dit pas**
   qu’il y a eu un déploiement : l’agent 2 doit le trouver seul avec `kubectl` et `git`
   (`--with-change-context` pour l’inclure comme le ferait l’agent 1 de l’architecture).
2. **Il lance l’agent 2** avec `--live-tools`, plus `--qdrant` (serveur Qdrant du compose) et
   `--llm` si demandés. Le rapport est dans `out/incidents/<incident>/`.

Avec `--capture` (ou `lab capture <fichier>.agent1.json`), l’incident réel est aussi enregistré :
les réponses de tous les outils deviennent un scénario rejouable hors ligne
(`evaluation/scenarios/Snn-lab-<panne>.json`, vraie cause tirée du `.truth.json`) et une fiche est
ajoutée au RAG (`knowledge/incidents_lab.json`, origine `lab_observation`). Ensuite, réindexer
Qdrant avec `knowledge_index`. Attention : un scénario de labo retrouve sa propre fiche dans le
RAG, donc son « RAG hit » est surévalué.

`--wait` attend que la panne soit visible (fuite : ≥ 6 min et mémoire ≥ 75 % ou un
redémarrage ; config : ≥ 2 redémarrages ; dépendance et pool : ≥ 5 % d’erreurs).

## Limites (à dire honnêtement)

- L’application et les pannes sont fabriquées, mais leurs effets sont réels : vraie mémoire,
  vrai OOMKill, vrai CrashLoopBackOff, vraies connexions refusées, vrais logs et métriques.
- Prometheus n’a que l’historique depuis `up` : la comparaison « il y a un jour » est vide.
- `gc_pause` (métrique Go) n’existe pas pour une app Python ; l’outil répond « pas de donnée ».
- Le labo est mono-nœud et petit : il ne teste ni la montée en charge ni un vrai réseau.
