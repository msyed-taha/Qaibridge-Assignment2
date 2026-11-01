# Qaibridge Assignment 2

An isolated, containerized version of the reduced Qaibridge educational simulator. It includes a circuit simulator, guided quantum challenges, and PostgreSQL-backed simulation history. The original Qaibridge project and Assignment 1 folder are not modified.

## Assignment deliverables

- `Dockerfile` builds the Part I application image.
- `docker-compose.yml` runs Part I's application and PostgreSQL database with persistent storage.
- `docker-compose-part2.yml` runs Part II from a source-code bind mount, with separate container names, ports, and database storage.
- `Jenkinsfile` tests the checked-out project in Docker, deploys Part II with Compose, and emails the commit author the build result.
- `docker/init/create-readonly-role.sh` creates a read-only database role on first database initialization.

Part I uses host port `8081` for the web application and host port `5432` for PostgreSQL, whose container listens on `5433`. This retains the standard external database port expected by the submission form while keeping host and container ports different. Part II uses host ports `8082` and `5434`; its containers and database volume are separate. Jenkins uses host port `8080`.

## Environment configuration

Copy `.env.example` to `.env` and set unique passwords before starting either Compose project. Use URL-safe alphanumeric passwords for all database roles; the app's `DATABASE_URL` is assembled by Compose. A secure hexadecimal value can be generated with `openssl rand -hex 24`. The admin password is only used to bootstrap the database; the web application uses a separate, non-superuser account.

Do not commit `.env`, paste credentials into chat, or include them in screenshots. On EC2, keep the deployment environment file at `/etc/qaibridge/assignment2.env` with permissions restricted to the administrators and Jenkins account.

## Part I: Docker Compose on EC2

Build and push the Part I image to the Docker Hub repository configured by `DOCKERHUB_IMAGE`. The `--platform linux/amd64` option is important when building on this Apple Silicon Mac because the EC2 host uses x86_64:

```sh
docker buildx build --platform linux/amd64 -t msyedtaha/qaibridge-assignment2:latest --push .
```

On EC2, clone this repository, configure its private environment file, and run:

```sh
docker compose --env-file /etc/qaibridge/assignment2.env -f docker-compose.yml pull
docker compose --env-file /etc/qaibridge/assignment2.env -f docker-compose.yml up -d
```

Part I is available at `http://EC2_PUBLIC_IP:8081`. The evaluator database connection uses EC2's public IP, port `5432`, database `qaibridge_assignment`, and the `qaibridge_readonly` credentials from the environment file. Do not grant the evaluator write access.

## Part II: Jenkins pipeline

Install Jenkins, Docker Engine/Compose, and the Jenkins Git, Pipeline, Docker Pipeline, GitHub, and Email Extension plugins on EC2. Configure the sender mailbox and SMTP credentials in Jenkins' system configuration; do not put email credentials in the repository. Configure a Pipeline job from this GitHub repository, enable the GitHub push trigger, and set its webhook to:

```text
http://EC2_PUBLIC_IP:8080/github-webhook/
```

Configure Jenkins SMTP with a sender mailbox credential. The pipeline uses Docker Pipeline to run the test suite inside `python:3.11-slim`, then starts Part II using the source bind mount in `docker-compose-part2.yml`. The Part II deployment is intentionally separate from Part I and can be stopped without interrupting Part I:

```sh
docker compose --env-file /etc/qaibridge/assignment2.env -f docker-compose-part2.yml down
```

The pipeline emails the latest Git commit author's address. For the assignment's collaborator-push check, commits pushed by a collaborator should use that collaborator's verified email address, matching the GitHub account email used for commits.

Part II reuses the published Part I application image for its installed Python dependencies, while the read-only bind mount overlays the checked-out source at `/app`. This avoids downloading dependencies during each deployment and keeps Part II independent of a Dockerfile build.

## Local verification

The Python test suite uses in-memory SQLite and does not require Docker or PostgreSQL:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt pytest httpx
python -m pytest -q
```

To run containers locally, install Docker Desktop, copy `.env.example` to `.env`, replace all placeholder passwords, then run either Compose file with `--env-file .env`.
