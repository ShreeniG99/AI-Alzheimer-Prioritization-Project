# AWS EC2 deployment

This is a research prototype deployment pattern with separate containers for:
- React frontend
- FastAPI backend
- PostgreSQL database

## 1. Create an EC2 instance

Use Ubuntu 24.04 LTS. Attach an EBS volume large enough for the database and model artifacts.

Security group:
- 22/TCP from your admin IP only
- 80/TCP from the internet if using HTTP
- 443/TCP from the internet if using HTTPS
- Do NOT expose PostgreSQL 5432 publicly.

For a demonstration, port 3000 can be exposed, but production should put Nginx/ALB in front of the application and use HTTPS.

## 2. Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker $USER
newgrp docker
```

## 3. Copy project

```bash
git clone YOUR_PRIVATE_REPOSITORY_URL
cd AI_Alzheimer_Prioritization_Project
```

Copy the trained outputs to the server (e.g. with `scp`), keeping the same folders:

- `artifacts/` (`<stage>_model.joblib`, `<stage>_metadata.json`, `stage1_bootstrap_predictions.npz`)
- `reports/prioritized_cohort.csv`
- `data/processed/stage1.csv` (needed for bootstrap uncertainty)

These contain ADNI-derived subject-level data. Keep them off public repositories,
restrict file permissions (`chmod -R go-rwx artifacts reports data`), and use an
encrypted EBS volume.

## 4. Configure secret

Create `.env` from `.env.example`:

```env
POSTGRES_PASSWORD=use-a-long-random-secret
JWT_SECRET=use-another-long-random-secret
ADMIN_USERNAME=admin
ADMIN_PASSWORD=use-a-strong-initial-password
CORS_ORIGINS=http://SERVER_PUBLIC_IP:3000
```

Generate secrets with `python3 -c "import secrets; print(secrets.token_urlsafe(48))"`.
The admin account is created on first start. Sign in, then create clinician accounts
(`POST /auth/users` or `docker compose exec backend python -m backend.init_db --create-user NAME`).

## 5. Start

```bash
docker compose up -d --build
docker compose ps
```

## 6. Verify

```bash
curl http://SERVER_PUBLIC_IP:8000/health
# {"status":"ok","available_stages":["stage1", ...]}
```

Open the frontend:
`http://SERVER_PUBLIC_IP:3000`

## Production hardening

For institutional use:
- HTTPS/TLS
- private database networking
- AWS Secrets Manager or equivalent
- IAM least privilege
- CloudWatch logs/metrics
- encrypted EBS and database backups
- ship the audit log (`audit_log` table) to central, append-only storage
- no PHI in frontend logs; patient references should be pseudonymous
- SSO / MFA in front of the built-in JWT login
- vulnerability scanning
- model versioning
- dataset/version provenance
- clinical validation and governance

Do not describe an EC2 demo as a clinically validated deployment.
