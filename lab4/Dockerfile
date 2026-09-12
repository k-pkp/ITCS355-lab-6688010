# ITCS355 Lab 1 — training image
#
# Base image pinned BY DIGEST. The tag `python:3.11-slim` is a moving pointer; this
# digest is the multi-arch index that the tag resolved to on 2026-09-07, and it will
# resolve to the same bytes forever. buildx still selects the linux/amd64 manifest from
# inside the index, so `--platform linux/amd64` keeps working.
#
# Re-resolve after a deliberate base upgrade:
#   docker pull python:3.11-slim && docker inspect --format='{{index .RepoDigests 0}}' python:3.11-slim
# The digest is written out on both FROM lines rather than carried in an ARG. An ARG
# reads better, but it hides the pin from anything that greps the FROM line — including
# the course's own grading script, which looks for `^FROM .*@sha256:`. A pin a checker
# cannot see is a pin you will be asked to prove by hand.
FROM python@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies first so this layer caches independently of your source.
COPY requirements.txt ./
# --require-hashes turns a silently-substituted package into a build failure. It also
# forces every requirement to be pinned, which is why requirements.txt is compiled from
# requirements.in rather than written by hand.
RUN pip install --require-hashes --prefix=/install -r requirements.txt


FROM python@sha256:9534e5a8e315485d4061ed659af0fd78a284c015f9b73661b41d6bab25604534 AS runtime

# Non-root. A training container has no reason to run as root, and graders check.
RUN useradd --create-home --uid 10001 runner
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    GIT_PYTHON_REFRESH=quiet

COPY --from=builder /install /usr/local
WORKDIR /app
# The runtime user needs to own its working directory: MLflow's sqlite backend
# creates its artifact root at ./mlruns, and WORKDIR makes /app root-owned.
RUN chown runner:runner /app
COPY --chown=runner:runner src/ ./src/
COPY --chown=runner:runner cloudlayer/ ./cloudlayer/
COPY --chown=runner:runner scripts/ ./scripts/

USER runner

# Credentials NEVER enter an image layer. They arrive at runtime from SECRET_STORE_PATH
# or from the platform's identity. If you find yourself adding an ARG for a key, stop.
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--n-estimators", "200", "--max-depth", "8"]
