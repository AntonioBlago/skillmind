# Design-Doc: FalkorDB-Backend + GraphRAG (als zusätzliche Option)

**Status:** Draft · **Autor:** Antonio Blago · **Datum:** 2026-06-04 · **Ziel-Version:** 0.4.0

> **Verifiziert am 2026-06-04** gegen lokalen Docker-Container `falkordb/falkordb`
> (Image-Digest `sha256:0befcaeb…`, Graph-Modul v41809, `vectorset`-Modul v1, SDK `falkordb==1.6.1`).
> Vektor-Index-Syntax, `vecf32`-Insert, `db.idx.vector.queryNodes` und Graph-Traversierung
> laufen end-to-end über das Python-SDK. **Wichtig:** `score` aus `queryNodes` ist die
> **Cosine-Distanz** (identisch → ~0) → im Store als `similarity = 1 - distance` umrechnen.

## 1. Ziel & Scope

FalkorDB als **zusätzliches Store-Backend** (Nr. 6) in SkillMind aufnehmen und darauf
**GraphRAG** als optionalen Retrieval-Modus anbieten.

- **Kein Ersatz:** Pinecone/Chroma/Qdrant/FAISS/Supabase bleiben unverändert. Default bleibt `chroma`.
- **Opt-in:** Wer FalkorDB will, setzt `store.backend = "falkordb"`. Wer zusätzlich GraphRAG
  will, schaltet `store.falkordb_graphrag = true`.
- **Drop-in-kompatibel:** Der FalkorDB-Store implementiert den vollen `MemoryStore`-Vertrag
  (`add`, `add_batch`, `query`, `get`, `update`, `delete`, `list_all`, `count`, `clear`),
  d.h. alles Bestehende (MCP-Tools, Trainer, YouTubeLearner) funktioniert ohne Änderung.
- **Migration:** Antonio zieht seine eigenen Memories von Pinecone nach FalkorDB (gehostet auf Railway).

### Warum FalkorDB der richtige Kandidat ist
FalkorDB ist die einzige Backend-Option, die **Vektor-Index UND Graph-Traversierung in einer
Engine** kann (Redis-basiert, Cypher-Abfragen, eingebauter Vektor-Index). Damit liefert ein
einziger Store:
- klassisches RAG über `query()` (Vektor-KNN) — kompatibel mit dem bestehenden Vertrag,
- GraphRAG über Multi-Hop-Traversierung des Wissensgraphen — der eigentliche Mehrwert.

## 2. Architektur-Einfügung

Bestehendes Plugin-Muster (`src/skillmind/store/`):

```
MemoryStore (base.py, ABC)
 ├── chroma_store.py
 ├── pinecone_store.py
 ├── qdrant_store.py
 ├── faiss_store.py
 ├── supabase_store.py
 └── falkordb_store.py   ← NEU
create_store() (store/__init__.py)  ← NEUER elif-Zweig "falkordb"
```

Änderungen am Kern: **minimal**.
1. `store/falkordb_store.py` — neue Klasse `FalkorDBStore(MemoryStore)`.
2. `store/__init__.py` — ein `elif backend == "falkordb": ...`.
3. `config.py` — FalkorDB-Felder in `StoreConfig` + env-Mapping in `resolve_env()`.
4. `pyproject.toml` — optionale Dependency-Gruppe `falkordb`.

## 3. Graph-Schema

Aus dem bestehenden `Memory`-Modell (id, type, topic, title, content, tags, source,
confidence, created_at, …) abgeleitet.

### Knoten
| Label | Properties | Quelle |
|---|---|---|
| `:Memory` | `id, type, topic, title, content, source, confidence, created_at, updated_at` + `embedding` (Vektor) | 1:1 aus `Memory` |
| `:Topic` | `name` | aus `Memory.topic` |
| `:Tag` | `name` | aus `Memory.tags[]` |
| `:Entity` | `name, kind` | Claude-Extraktion aus `content` (GraphRAG-Build) |

### Kanten
| Beziehung | Bedeutung |
|---|---|
| `(:Memory)-[:HAS_TOPIC]->(:Topic)` | deterministisch aus `topic` |
| `(:Memory)-[:HAS_TAG]->(:Tag)` | deterministisch aus `tags[]` |
| `(:Memory)-[:MENTIONS]->(:Entity)` | aus Entity-Extraktion |
| `(:Memory)-[:RELATES_TO {weight}]->(:Memory)` | aus `[[name]]`-Links im Content + Embedding-Similarität |

> Die `[[name]]`-Verlinkung aus dem Memory-Schema (siehe Workspace-Memory-Konvention) wird
> direkt zu `RELATES_TO`-Kanten — der Graph ist also teils schon im Content kodiert.

### Indizes
- **Vektor-Index** auf `:Memory(embedding)` — Dimension `384` (`all-MiniLM-L6-v2`, aus `EmbeddingConfig.dimension`), Distanz `cosine`.
- **Range/Exact-Index** auf `:Memory(id)`, `:Topic(name)`, `:Tag(name)`, `:Entity(name)` für schnelle Lookups & MERGE.

## 4. Retrieval-Flows

### 4a. RAG (Default, `falkordb_graphrag = false`)
Reiner Vektor-KNN — verhält sich exakt wie Pinecone & Co.:
```cypher
CALL db.idx.vector.queryNodes('Memory', 'embedding', $k, $queryVec)
YIELD node, score
RETURN node, score
```
→ Top-k `Memory`-Knoten → `QueryResult`-Liste. Erfüllt den `query()`-Vertrag 1:1.

### 4b. GraphRAG (opt-in, `falkordb_graphrag = true`)
1. **Seed:** Vektor-KNN wie oben → Top-`seed_k` Memories (z.B. 5).
2. **Expand:** 1–2 Hops über `HAS_TOPIC` / `HAS_TAG` / `MENTIONS` / `RELATES_TO`
   sammelt verbundene Memories (gemeinsames Thema, geteilte Entitäten, explizite Links).
3. **Re-Rank:** Kandidaten kombiniert scoren = Vektor-Similarität × Confidence × Graph-Nähe
   (Hop-Distanz-Abschlag, Kantengewicht).
4. **Assemble:** Top-`limit` als `QueryResult` zurück — mehr Kontext, weil thematisch/relational
   verbundenes Wissen mitkommt, das reines Top-k verpasst.

```cypher
// vereinfachte Expansion (Hop 1–2 ab den Seed-IDs)
MATCH (seed:Memory) WHERE seed.id IN $seedIds
MATCH (seed)-[:HAS_TOPIC|HAS_TAG|MENTIONS|RELATES_TO*1..2]-(rel:Memory)
WHERE rel.id <> seed.id
RETURN DISTINCT rel, seed.id AS via
LIMIT $expandLimit
```

GraphRAG-Schalter wird in `query()` ausgewertet; ohne Schalter = Pfad 4a. Damit ändert sich
für bestehende Aufrufer nichts.

## 5. Config-Erweiterung

In `StoreConfig` (config.py) ergänzen:
```python
# FalkorDB (Graph + Vector)
falkordb_url: str = Field(default="redis://localhost:6379", description="FalkorDB/Redis URL")
falkordb_password: str = Field(default="", description="FalkorDB password (requirepass)")
falkordb_graph: str = Field(default="skillmind", description="Graph name")
falkordb_graphrag: bool = Field(default=False, description="Enable GraphRAG multi-hop retrieval")
falkordb_seed_k: int = Field(default=5, description="GraphRAG: vector seeds before graph expansion")
falkordb_hops: int = Field(default=2, description="GraphRAG: max traversal hops (1-2)")
```
`backend`-Description um `| falkordb` erweitern.

In `resolve_env()` ergänzen:
```python
"FALKORDB_URL": "store.falkordb_url",
"FALKORDB_PASSWORD": "store.falkordb_password",
```
(`SKILLMIND_BACKEND` existiert bereits → erlaubt `backend=falkordb` per env.)

## 6. Dependencies

`pyproject.toml`:
```toml
falkordb = ["falkordb>=1.0"]   # offizielles Python-SDK (spricht Redis-Protokoll + Cypher)
```
und in die `all`-Gruppe aufnehmen. Entity-Extraktion für den Graph-Build nutzt das bereits
vorhandene `anthropic`-Paket — keine neue schwere Abhängigkeit.

## 7. Migrationspfad Pinecone → FalkorDB

Neuer CLI-Befehl (oder Einmal-Skript): `skillmind migrate --from pinecone --to falkordb`.

1. **Quelle lesen:** `pinecone_store.list_all(limit=10000)` → alle `Memory`-Objekte.
2. **Ziel befüllen:** `falkordb_store.add_batch(memories)` — re-embedded via `EmbeddingEngine`,
   legt `:Memory`-Knoten + deterministische `HAS_TOPIC`/`HAS_TAG`-Kanten an.
3. **Graph-Build (für GraphRAG):** einmaliger Pass
   - `[[name]]`-Links im Content → `RELATES_TO`-Kanten,
   - optional Claude-Entity-Extraktion → `:Entity` + `MENTIONS`,
   - optional Embedding-Similarität > Schwelle → zusätzliche `RELATES_TO {weight}`.
4. **Verify:** `count()` Quelle vs. Ziel; Stichproben-`query()` auf beiden vergleichen.
5. **Umschalten:** `store.backend = "falkordb"` in der Config / `SKILLMIND_BACKEND=falkordb`.
   Pinecone bleibt als Fallback erhalten (kein Datenverlust).

## 8. Hosting

### Dev (lokal, Spike) — verifiziert
```bash
docker run -d --name falkordb \
  -p 6379:6379 -p 3000:3000 \
  -v falkordb_data:/data \
  -e REDIS_ARGS="--requirepass skillmind-dev" \
  falkordb/falkordb
```
Port 3000 = eingebaute Browser-UI. Config: `falkordb_url = redis://:skillmind-dev@localhost:6379`.

> **Achtung:** Das Passwort MUSS über die Env-Variable `REDIS_ARGS` gesetzt werden.
> Ein positionales `--requirepass …` nach dem Image-Namen wird vom FalkorDB-Entrypoint
> **ignoriert** (Server startet dann ohne Auth). Dasselbe `REDIS_ARGS`-Muster gilt auf Railway.

### Prod (Railway — Antonios Ziel)
1. Service aus Docker-Image `falkordb/falkordb` deployen.
2. **Volume** auf `/data` mounten (sonst Datenverlust bei Redeploy).
3. **TCP-Proxy** aktivieren — FalkorDB spricht Redis über rohes TCP (Port 6379), kein HTTP.
   Railway liefert dafür eine öffentliche `host:port`-Adresse.
4. **Passwort Pflicht:** Redis mit `--requirepass <stark>` starten (Start-Command/Redis-Args).
5. MCP-Server konfigurieren: `FALKORDB_URL=redis://:<pw>@<railway-host>:<port>`,
   `SKILLMIND_BACKEND=falkordb`.
6. **Security:** Redis-Port nie unauthentifiziert offen — Railway Private Networking bevorzugen,
   wo möglich; öffentlicher TCP-Proxy nur mit gesetztem Passwort.

| | Dev lokal | Railway (Prod) |
|---|---|---|
| Setup | trivial | managed Container, TCP-Proxy nötig |
| Always-on | nein | ja |
| Kosten | 0 € | ~5 $/Monat (Hobby, nutzungsbasiert) |

## 9. Implementierungs-Checkliste

| # | Schritt | Aufwand | Status |
|---|---|---|---|
| 1 | `StoreConfig`-Felder + `resolve_env()`-Mapping | S | ✅ |
| 2 | `pyproject.toml`: `falkordb`-Extra | XS | ✅ |
| 3 | `FalkorDBStore`: Vertrag (add/query/get/update/delete/list_all/count/clear) + Vektor-Index | L | ✅ |
| 4 | `create_store()`: `elif "falkordb"` | XS | ✅ |
| 5 | GraphRAG-Pfad in `query()` (Seed → Expand → Re-Rank) hinter `falkordb_graphrag` | M | ✅ (Inverse-Degree-Gewichtung gegen Hub-Problem) |
| 6 | Graph-Build (`build_graph`: `[[wikilinks]]` + semantische `RELATES_TO`) | M | ✅ |
| 7 | Migrationsbefehl `migrate --from pinecone --to falkordb` | M | ✅ |
| 8 | Tests (s.u.) | M | ✅ |
| 9 | Docs: README-Backend-Tabelle + `reference_skillmind_usage.md` | S | ✅ |

## 10. Tests

- **Unit (offline):** Cypher-Builder, Re-Rank-Scoring, `[[name]]`-→-`RELATES_TO`-Parsing —
  ohne laufende DB, reine Logik.
- **Integration (opt-in):** gegen lokalen Docker-FalkorDB; `@pytest.mark.skipif` wenn
  `FALKORDB_URL` nicht gesetzt → CI bleibt grün ohne DB.
- **Vertrags-Parität:** dieselbe Test-Suite, die Pinecone/Chroma gegen `MemoryStore` prüft,
  auch gegen `FalkorDBStore` laufen lassen (add→query→get→update→delete Roundtrip).
- **GraphRAG vs RAG:** ein Seed-Set, einmal mit/ohne `falkordb_graphrag` → GraphRAG liefert
  ≥ die RAG-Treffer plus verbundene Memories.

## 11. Risiken & offene Punkte
- ✅ **ERLEDIGT:** FalkorDB-Vektor-Index-Syntax gegen aktuelle Version verifiziert
  (`CREATE VECTOR INDEX … OPTIONS {dimension, similarityFunction}`, `vecf32(...)`,
  `db.idx.vector.queryNodes`). SDK `falkordb==1.6.1`.
- ✅ **ERLEDIGT:** `score` = Cosine-Distanz → `similarity = 1 - distance` (im Store umrechnen).
- ✅ **ERLEDIGT:** Passwort via `REDIS_ARGS`-Env, nicht positionales `--requirepass`.
- ✅ **ERLEDIGT (verifiziert 2026-06-04):** Ein **persistierter** Vektor-Index reindiziert
  Knoten, die nach einem `MATCH (n) DETACH DELETE n` neu eingefügt werden, NICHT zuverlässig
  (Index meldet `OPERATIONAL`, `queryNodes` liefert aber 0 Treffer). Normaler Insert über eine
  neue Verbindung **ohne** vorherigen Bulk-Delete funktioniert dagegen einwandfrei. Fix:
  `clear()` droppt den Index (`DROP VECTOR INDEX …`) und legt ihn neu an, damit der Store
  danach garantiert abfragbar bleibt.
- Embedding-Dimension muss zwischen Index-Anlage und `EmbeddingConfig` konsistent sein (384).
- Entity-Extraktion via Claude kostet Tokens/Zeit → nur beim Graph-Build, nicht bei jeder Query;
  ggf. als separater, nachgelagerter Schritt (Konsolidierung).
- `falkordb`-SDK zieht `redis>=8` als Dependency — im Server-Venv bereits installiert.

## 12. Verifizierte Cypher-Bausteine (Referenz für die Implementierung)
```cypher
-- Index (einmalig bei initialize())
CREATE VECTOR INDEX FOR (m:Memory) ON (m.embedding)
  OPTIONS {dimension: 384, similarityFunction: 'cosine'};

-- Upsert eines Memory + deterministische Topic-Kante (add/update)
MERGE (m:Memory {id:$id}) SET m.title=$title, m.embedding=vecf32($vec)
MERGE (t:Topic {name:$topic}) MERGE (m)-[:HAS_TOPIC]->(t);

-- RAG: Vektor-KNN (query, Distanz → 1-distance als score)
CALL db.idx.vector.queryNodes('Memory','embedding',$k, vecf32($v))
  YIELD node, score RETURN node, score;

-- GraphRAG: 1–2 Hop-Expansion ab Seed-IDs
MATCH (seed:Memory) WHERE seed.id IN $seedIds
MATCH (seed)-[:HAS_TOPIC|HAS_TAG|MENTIONS|RELATES_TO*1..2]-(rel:Memory)
WHERE rel.id <> seed.id RETURN DISTINCT rel LIMIT $expandLimit;
```
