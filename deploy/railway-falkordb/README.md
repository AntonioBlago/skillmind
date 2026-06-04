# FalkorDB on Railway — SkillMind production store

Deploys the FalkorDB graph+vector database that backs the `falkordb` SkillMind store
backend. Pinecone stays the default/fallback; FalkorDB is the opt-in, GraphRAG-capable
target you migrate memories into.

> **Secrets stay local.** Your password, the TCP host/port and the full connection URL
> belong in `connection.local.env` (gitignored). Never commit instance values to a
> public repo. This guide uses placeholders like `<password>` and `<host>:<port>`.

## Prerequisites

- A [Railway](https://railway.app) account and the CLI (`railway --version`, logged in via `railway login`).
- Run every command from this folder so it stays linked to your project + service.

## 1. Create the project and service

```bash
cd deploy/railway-falkordb

# Generate a strong password and keep it in connection.local.env
PW="$(python -c 'import secrets; print(secrets.token_urlsafe(24))')"

railway init -n skillmind-falkordb

# FalkorDB from the public image, password injected via REDIS_ARGS
railway add --service falkordb --image falkordb/falkordb:latest \
    --variables "REDIS_ARGS=--requirepass $PW"

railway service falkordb         # link the service for the next commands
```

**Security:** the password MUST be passed via the `REDIS_ARGS` env var. A positional
`--requirepass` after the image name is ignored by the entrypoint, which would leave
the server open. The Redis/6379 port must never be reachable without auth.

## 2. Persistent volume + public TCP endpoint

```bash
# Persistent storage on /data (Git Bash mangles the path → MSYS_NO_PATHCONV=1)
MSYS_NO_PATHCONV=1 railway volume add -m /data
railway volume list                 # confirm it attached to falkordb

# TCP proxy so SkillMind can reach Redis on 6379 from the internet
railway domain -p 6379              # prints <region>.proxy.rlwy.net:<port>
#   If the CLI only makes an HTTP domain, enable the TCP Proxy in the dashboard:
#   falkordb service -> Settings -> Networking -> TCP Proxy -> target port 6379

railway logs                        # wait for "Ready to accept connections"
```

Record the result in `connection.local.env`:

```
FALKORDB_URL=redis://:<password>@<region>.proxy.rlwy.net:<port>
```

## 3. Smoke test

```bash
python -c "from falkordb import FalkorDB; from urllib.parse import urlparse; \
u=urlparse('<FALKORDB_URL>'); \
d=FalkorDB(host=u.hostname,port=u.port,password=u.password); print('ping', d.connection.ping())"
```

## 4. Migrate memories from another backend

The migration shares one embedding engine across source and target, so both MUST use
the same embedding model/dimension (re-embedding happens in the target). IDs are
preserved, so the migration is idempotent and safe to re-run.

Create a `migrate-config.yml` carrying the source backend credentials AND the Railway
`falkordb_url` (keep it local — it holds secrets), then:

```bash
# Dry run — counts only, writes nothing
python -m skillmind.cli.main -c migrate-config.yml migrate --from pinecone --to falkordb --dry-run

# Real migration (idempotent; the source is only read, never modified)
python -m skillmind.cli.main -c migrate-config.yml migrate --from pinecone --to falkordb -y
```

The command prints `source ... holds N`, `migrated: N`, `target (falkordb) now holds N`.
