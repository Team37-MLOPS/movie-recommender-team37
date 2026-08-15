# AWS EC2 Demo Deployment

This project can be hosted on AWS with a free-tier strict posture by running the Docker Compose stack on one small EC2 instance in `us-east-1`.

## Cost Guardrails

1. Create an AWS Budget before launching resources.
2. Use one small Ubuntu EC2 instance for demos.
3. Stop the EC2 instance when not in use.
4. Avoid MWAA, EMR, EKS, RDS, and always-on managed services for the initial coursework demo.
5. Keep Prometheus and Grafana access temporary or restricted to your IP.

## EC2 Setup

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-plugin git
sudo usermod -aG docker ubuntu
```

Log out and back in, then clone the repository:

```bash
git clone https://github.com/Team37-MLOPS/movie-recommender-team37.git
cd movie-recommender-team37
cp .env.aws.example .env
docker compose up --build
```

## Security Group

Open only the ports needed for a demo, restricted to your IP:

- `22`: SSH
- `8000`: FastAPI
- `8080`: Airflow
- `5000`: MLflow
- `3000`: Grafana

Avoid exposing Postgres and Prometheus publicly.

## Optional S3 DVC Remote

```bash
dvc remote add -d aws-s3 s3://replace-with-your-bucket/dvc
dvc push
```

Use IAM credentials with the minimum permissions needed for the DVC bucket.
