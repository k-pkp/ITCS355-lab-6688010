# ITCS355 — Labs 1 to 5

Student project `itcs355-6688010`. Predicting machine failure within 7 days from sensor
readings, built up across five labs: reproducible training, tracking and registry, serving
and rollback, CI/CD and drift, then cloud MLOps, cost and an LLM step.

**One folder per lab. Each is self-contained** — `cd lab3 && make test` works without
reaching outside that folder. The shared modules (`src/`, `cloudlayer/`, `scripts/`,
`tests/`) are duplicated into all five on purpose, so that a lab can be read, run and graded
on its own.

| Lab | Folder | What it delivers | Its evidence |
|:--|:--|:--|:--|
| 1 | [`lab1/`](lab1) | Reproducible training container | [README claim](lab1/README.md) · [`Dockerfile`](lab1/Dockerfile) · [`requirements.txt`](lab1/requirements.txt) |
| 2 | [`lab2/`](lab2) | Budgeted study, verified prices, registry lineage | [`lab2-comparison.md`](lab2/reports/lab2-comparison.md) |
| 3 | [`lab3/`](lab3) | Service, load matrix, canary and rollback | [`lab3-load.md`](lab3/reports/lab3-load.md) · [`lab3-canary.md`](lab3/reports/lab3-canary.md) |
| 4 | [`lab4/`](lab4) | CI/CD, drift, the [blocked bad commit](https://github.com/k-pkp/ITCS355-lab-6688010/pull/1) | [`lab4-drift.md`](lab4/reports/lab4-drift.md) · [`lab4-postmortem.md`](lab4/reports/lab4-postmortem.md) · [`lab4-blocked-commit.md`](lab4/reports/lab4-blocked-commit.md) |
| 5 | [`lab5/`](lab5) | Pipeline gate, cost, portability, LLM gate | [`lab5-cost.md`](lab5/reports/lab5-cost.md) · [`lab5-portability.md`](lab5/reports/lab5-portability.md) · [`lab5-llm.md`](lab5/reports/lab5-llm.md) |

## Grading

The course's mechanical scripts expect to run at a repository root. With one folder per lab
they need to run **inside a lab folder** instead:

```bash
git clone <this repo> && cd ITCS355-lab-6688010

cd lab1 && bash /path/to/instructor/grade_lab1.sh "file://$PWD" ; cd ..
cd lab2 && bash /path/to/instructor/grade_lab.sh 2 "file://$PWD" ; cd ..
```

Or check one lab by hand, from inside its folder:

```bash
make data && make test          # dataset contract and split property tests
make reproduce && make verify   # the one command, and the claim check
make portability-audit
```

`make` needs a Python 3.11 environment; the lock file targets 3.11 and the container builds
one for itself, so `make reproduce` is unaffected by the host interpreter.

Two things live outside the lab folders because they cannot work anywhere else:
`.github/workflows/` (GitHub only runs workflows found at the repository root — `lab4/`
keeps an identical readable copy) and `.gitignore`.

## The answers that are graded as prose

Each lab's README carries its own reasoning. These four are asked for explicitly and are
gathered here so they are not buried.

### Which pinning would you drop first under time pressure

**The hashes.** Versions are still pinned without them, so the build changes only if a
maintainer republishes an artifact under an existing version — rare, and loud. What breaks is
the guarantee against a *substituted* wheel, which is a security regression rather than a
reproducibility one. Dropping the base-image digest is worse: `python:3.11-slim` moves weekly
and takes glibc and OpenSSL with it. Dropping seeds is worst — the number changes on the same
machine, on the same day, and nothing tells you.

### Which incident each data contract test would have caught

| Test | The incident |
|:--|:--|
| `test_schema_columns_present_and_typed` | The gateway's CSV export gains a column after a firmware update, or `machine_id` starts arriving zero-padded as a string. Training then one-hot encodes an identifier, or drops a feature, and the metric barely moves. |
| `test_no_nulls_in_required_columns` | A polling failure leaves `vibration_mm_s` empty for some machines. Nulls become imputed means, the most important feature turns into a constant for those rows, and they all score the same. |
| `test_features_within_plausible_ranges` | A recalibration ships and `temp_c` arrives 6 °C high — the incident reconstructed in Lab 4's post-mortem. Ranges catch the gross version at ingest, a week before delayed labels would. |
| `test_target_is_binary_and_not_degenerate` | The labelling job breaks and writes 0 for everything. Training succeeds, ROC AUC is 0.5, and a model that predicts "never fails" ships. |
| `test_identifier_is_unique` | An upstream join fans out and duplicates readings. Duplicates cross the train/test boundary and the reported score becomes memorisation. |
| `test_no_machine_leaks_across_splits` | Someone changes the split to row-wise "because it is simpler". Validation jumps, everyone is pleased, and production performance does not move. |

### Who should be allowed to promote a model, and on what evidence

Not the person who trained it, and not automatically. Promotion to production should require
sign-off from whoever carries the pager for the service, because promotion is the moment a
model becomes someone's operational burden. The evidence they should require: the eight
lineage fields on the registered version; the gate's output showing the margin exceeds seed
noise; the canary result with its detection time; and a named rollback path somebody has
actually executed. "The metric is higher" is not evidence — a 0.005 improvement on this data
is indistinguishable from a different random split. CI may promote to **staging** unattended.
Nothing promotes to production unattended.

### What should trigger a retrain

| Strategy | When it is right here | How it fails here |
|:--|:--|:--|
| **Fixed schedule** (chosen: weekly, Sun 02:00) | The fleet changes slowly and the label takes 7 days, so a week of new data moves the training set by under 1% — well inside seed noise. Weekly is chosen for operational rhythm: a pipeline that runs regularly is one that still works when you need it. | Wastes compute re-deriving the same model, and a schedule that always succeeds trains you to stop reading its output. It cannot react: a Monday change leaves the model six days stale. |
| **Data volume threshold** | Would suit a growing fleet — retrain when 20% more machines have a full history. Ties work to information gained. | This fleet is ~240 machines and roughly fixed, so it would fire almost never, and a retraining path that runs once a year is broken without anyone knowing. |
| **Drift-triggered** | The instinctive answer, and the dangerous one. | Two independent failures. The detector is blind to the drift most likely to happen — a fleet composition change scored PSI 0.0087, below the noise floor, while a harmless uniform shift scored 0.383. And a drift alert cannot distinguish a real change from a broken upstream pipeline: the 6 °C shift in Lab 4 was a recalibration bug, and retraining on it would have baked that into the model while replacing the last good one. |

**Chosen: fixed weekly schedule, with drift as an alert to a human, never as a trigger.** The
worst case for that is bounded and slow — the world changes on a Monday and the model is six
days stale. The worst case for the alternative is unbounded and fast: an upstream pipeline
breaks at 01:00, drift fires at 01:30, the automated retrain finishes at 02:00 on corrupted
data, and the working model is gone. One is recoverable by waiting; the other only by
noticing.

## Finishing on GCP

Everything below is implemented and unexercised. This is the full sequence, in order; each
step's output is what the next one needs. Run it from inside a lab folder — `lab5/` has all
of it. Project `itcs355-6688010`, region `asia-southeast1`, budget 800 THB.

**Costs, so nothing is a surprise:** the training jobs are a few THB. The endpoint is
8.889 THB/hour *from the moment it exists until it is deleted*, whether or not anything
calls it — 640 THB over a weekend, 80% of the term budget. Delete it the same day.

```bash
# 0. Tools and login, once per machine.
#    The adapter mints its own OAuth token, so gcloud is needed for administration only.
gcloud auth login
gcloud auth application-default login          # what the Python SDKs read
gcloud config set project itcs355-6688010
pip install google-cloud-aiplatform            # imported lazily by the adapter

# 1. Enable the one API everything below needs. Nothing works before this.
gcloud services enable aiplatform.googleapis.com
gcloud services list --enabled | grep aiplatform

# 2. Three scoped identities (Lab 5 Task 2). Create them, bind only what each needs.
for SA in itcs355-train itcs355-serve itcs355-ci; do
  gcloud iam service-accounts create $SA --display-name "ITCS355 $SA"
done
PROJECT=itcs355-6688010

# training: read data, write artifacts and metrics. No deploy, no registry write.
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:itcs355-train@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:itcs355-train@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

# serving: read one artifact, write logs and metrics. Deliberately no storage write.
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:itcs355-serve@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:itcs355-serve@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/monitoring.metricWriter"

# CI: push images, run the pipeline, deploy, and act as the two above.
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:itcs355-ci@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding $PROJECT \
  --member="serviceAccount:itcs355-ci@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
gcloud iam service-accounts add-iam-policy-binding \
  itcs355-train@$PROJECT.iam.gserviceaccount.com \
  --member="serviceAccount:itcs355-ci@$PROJECT.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# 3. REDO the least-privilege experiment properly (this is the graded half of Task 2).
#    Scope arguments are ignored on a user credential; impersonation is a real boundary.
gcloud projects add-iam-policy-binding $PROJECT \
  --member="user:$(gcloud config get-value account)" \
  --role="roles/iam.serviceAccountTokenCreator"
gcloud storage cp README.md gs://itcs355-6688010/itcs355/probe.txt \
  --impersonate-service-account=itcs355-serve@$PROJECT.iam.gserviceaccount.com
#    Expect a 403. Record the exact message — it is what Drill 5 asks for.

# 4. Workload Identity Federation, so CD needs no stored key (Lab 4 Task 2).
gcloud iam workload-identity-pools create github --location=global \
  --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc itcs355 \
  --location=global --workload-identity-pool=github \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository=='k-pkp/ITCS355-lab-6688010'"
PROJECT_NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
gcloud iam service-accounts add-iam-policy-binding \
  itcs355-ci@$PROJECT.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attributes/repository/k-pkp/ITCS355-lab-6688010"
gcloud iam workload-identity-pools providers describe itcs355 \
  --location=global --workload-identity-pool=github --format='value(name)'
#    Paste that name into the GCP_WIF_PROVIDER repository secret.

# 5. Managed training (Lab 2 Task 1). Expect the FIRST submission to fail on permissions:
#    the identity that submits is yours, the one that runs is the service account.
#    Record which permission was missing — Drill 2 asks.
python -c "
from src import config
from cloudlayer.factory import get_adapter
adapter = get_adapter(config.load())
job = adapter.submit_training(
    open('reports/image-digest.txt').read().strip(),
    {'seed': 20260101, 'machine_type': 'e2-standard-4',
     'service_account': 'itcs355-train@itcs355-6688010.iam.gserviceaccount.com'})
print('job:', job)
print(adapter.wait_training(job))"

# 6. Register from the managed run, then deploy and smoke-test (Labs 2 and 3).
python scripts/register_model.py --stage Staging --training-job-id "<job id from step 5>"
python -c "
from src import config
from cloudlayer.factory import get_adapter
adapter = get_adapter(config.load())
endpoint = adapter.deploy('<model resource name>', 'itcs355-staging', 'n1-standard-4')
print('endpoint:', endpoint)
print(adapter.invoke(endpoint, {'temp_c':78.4,'vibration_mm_s':3.1,'pressure_kpa':315.2,
      'hours_since_service':4200.0,'load_pct':68.0,'ambient_humidity':55.0}))"

# 7. Record the LLM fixture live, so token counts come from usage fields (Lab 5 Task 9).
make llm-record
make llm-eval-mine          # confirm the gate still passes on the real recording

# 8. Schedule the pipeline (Lab 5 Task 3), weekly, matching pipeline/pipeline.yaml.
gcloud scheduler jobs create http itcs355-retrain \
  --location=asia-southeast1 --schedule="0 2 * * 0" --time-zone="Asia/Bangkok" \
  --uri="https://asia-southeast1-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/asia-southeast1/pipelineJobs" \
  --oauth-service-account-email=itcs355-ci@$PROJECT.iam.gserviceaccount.com

# 9. TEAR DOWN THE SAME DAY, then verify twice — deletion is asynchronous.
make teardown
python scripts/teardown_verify.py --lab 3
gcloud ai endpoints list --region=asia-southeast1        # must be empty
gcloud ai models list --region=asia-southeast1
gcloud scheduler jobs list --location=asia-southeast1    # a surviving schedule keeps billing
#    Re-run teardown_verify 24 hours later, then screenshot the empty console list:
#    that screenshot is a Lab 5 deliverable.

# 10. Close the loop on the cost report with the real number, filtered by tag.
make cost EST=60 ACT=<actual from billing> RPS=239.7 INSTANCE=e2-standard-4
```

After steps 5 to 7, three documents need their measured figures swapped for billed ones:
`lab2/reports/lab2-comparison.md` (modelled cost becomes job cost),
`lab5/reports/lab5-cost.md` (the estimate-versus-actual gap), and
`lab5/reports/lab5-llm.md` (recorded token counts). Each already says which of its numbers
is provisional, so the edits are small and located.

## Honest status

Real and exercised: object storage (GCS), the container registry (Artifact Registry — the
image is pushed and digest-pinned), DVC push, and Cloud Monitoring (`emit_metric` wrote a
real time series). Labs 2 and 3 were measured locally, priced at verified `asia-southeast1`
rates.

Implemented but not yet run: everything that needs the Vertex AI API — managed training,
managed endpoints, the managed LLM, and the Vertex teardown path. Each report says so where
it matters rather than implying otherwise, and `lab5/README.md` lists what changes once the
API is enabled.
