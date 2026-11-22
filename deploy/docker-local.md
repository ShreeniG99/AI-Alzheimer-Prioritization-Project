# Running IDEA with Docker on a laptop

Three containers: **PostgreSQL** (database), **backend** (FastAPI + models)
and **frontend** (React served by nginx).

## Before you start

1. Install **Docker Desktop** and start it. On Windows it needs WSL 2, and it
   asks for a restart the first time.
2. Get the project:
   ```powershell
   git clone https://github.com/ShreeniG99/AI-Alzheimer-Prioritization-Project.git alz-app
   cd alz-app
   git checkout shreenidhi/ui-refresh
   ```
3. Unzip the trained models into the project folder, so `artifacts\`,
   `reports\` and `data\processed\` exist. Models contain ADNI-derived data:
   they are mounted into the backend read-only and never built into an image.

## Start

```powershell
copy .env.example .env
notepad .env        # change every password and the JWT secret
docker compose up --build
```

The first build takes about 5–10 minutes. Later starts take seconds.

Open **http://localhost:3000** and sign in with the `ADMIN_USERNAME` /
`ADMIN_PASSWORD` from `.env`.

## Useful commands

| Command | What it does |
|---|---|
| `docker compose ps` | Show the three running containers |
| `docker compose logs -f backend` | Follow the backend log |
| `docker compose down` | Stop everything (data is kept) |
| `docker compose down -v` | Stop and delete the database |
| `docker compose up --build` | Rebuild after a code change |

## Checks

- http://localhost:8000/health lists four `available_stages` and four `tempo_stages`.
- If it lists none, the models were not unzipped into the project folder.
