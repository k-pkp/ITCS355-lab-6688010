# Lab 5 Task 4 — the portability test, and Task 2 — least privilege

## Part A — the audit

```
$ make portability-audit
PORTABILITY AUDIT PASSED — src, service, monitoring, tests contain no provider-specific strings
```

Clean. That result is worth exactly as much as the next section says it is.

## Part B — the swap

```
$ python scripts/portability_swap_check.py --second-provider local --endpoint http://127.0.0.1:8081
primary   gcp
secondary local

  [PASS] upload
  [PASS] download
  [PASS] invoke
```

The second adapter is `LocalAdapter`: `upload` and `download` against the filesystem,
`invoke` over HTTP against the same serving image that would sit behind a managed endpoint.
The same `Config` object, a different adapter, no change to any caller.

**Why the second adapter is not AWS.** `AwsAdapter.upload`, `download` and `invoke` are
implemented — S3 `upload_file`/`download_file` and a SageMaker runtime `invoke_endpoint`
that unwraps the response — but there is no AWS account behind them, so they have never
run. Code that has never executed is not evidence, and this course is explicit about that.
What is being proved with `local` is the thing the seam is actually for: that the caller's
code does not change when the implementation does. What is *not* proved is that the AWS
implementation is correct, and that distinction is stated rather than blurred.

## Part C — the write-up

**Which method was hardest, and why: `invoke`, and it is not close.** `upload` and
`download` are the same operation everywhere — bytes in, bytes out, addressed by a string.
`invoke` is where the providers stop agreeing about what a prediction *is*. Vertex returns
`{"predictions": [...]}` and unwraps to a list; SageMaker returns the container's raw body,
so what comes back is whatever `service/app.py` wrote; Azure ML returns the scoring script's
return value, which is a third shape again. The adapter can flatten all three to
`{"probability": ..., "model_version": ...}`, and it does, but every one of those unwrappings
is a guess about a response shape that the provider is free to change.

**Where the abstraction genuinely leaked, and could not be hidden.**

1. **Compilation returns different kinds of thing.** `compile_gcp` returns a *path*, because
   Vertex pipelines are compiled to a file and submitted. SageMaker's compiler would return a
   `Pipeline` *object*. There is no honest common return type — one is an artifact, the other
   is a live handle — so `compile_for` has a signature that is a lie in one direction or the
   other. It is documented in `cloudlayer/pipelines.py` rather than papered over.

2. **The audit passes and the leak is elsewhere.** `make portability-audit` scans `src/`,
   `service/`, `monitoring/` and `tests/` for provider strings. `gs://itcs355-6688010/itcs355`
   sits in `.dvc/config`, and `asia-southeast1-docker.pkg.dev/...` sits in `cloud.env`. Both
   are outside the audit's scope, both are provider-specific, and switching providers means
   editing both. The audit proves the *code* is neutral; it says nothing about the
   configuration, and configuration is where the real migration work turns out to live. This
   is precisely the failure mode the lab handout lists as "portability audit clean but the
   swap fails".

3. **Route paths.** SageMaker requires `/ping` and `/invocations` and will not be configured
   otherwise. The image cannot be renamed to suit it without becoming un-deployable
   elsewhere, so `cloudlayer/routes.py` records the difference and the adapter applies a path
   rewrite at deploy time — for AWS only. The abstraction absorbs the difference; it does not
   remove it, and one provider needs a component the other two do not.

**What a full migration would actually cost.** Honestly, and counting only what has been
built here:

| work | days |
|:--|--:|
| `upload`, `download`, `push_image` against S3 and ECR | 0.5 |
| `submit_training` / `wait_training` on SageMaker, including the run-time vs submit-time identity fight | 2 |
| `register_model` on model package groups, with approval status | 1 |
| `deploy` / `invoke` on real-time endpoints, plus the `/ping` and `/invocations` rewrite | 1.5 |
| `emit_metric` on CloudWatch, and rebuilding the dashboard | 1 |
| `compile_aws` for SageMaker Pipelines, including the ConditionStep / FailStep asymmetry | 2 |
| CI identity federation, IAM roles, and the first three permission failures | 2 |
| Re-verifying every price and rewriting the cost report | 1 |
| **Total** | **11 days** |

Then double it, because that estimate contains no time for the things that only appear once
real traffic hits a real endpoint. **Call it four to five weeks of one engineer.**

**Was the abstraction worth building? For this system, no — and yes for the one lesson.**

The honest accounting: the seam cost perhaps a day and a half of the work in this repo, and
it has bought exactly one thing so far, a swap test against a filesystem. If the goal were
to ship this system, that day and a half would have been better spent on the drift detector's
blind spot — it cannot see a fleet composition change at all, which is the drift most likely
to actually occur. Choosing GCP deliberately and writing `google.cloud.storage` directly into
`src/` would have produced a smaller, clearer codebase, and the migration estimate above says
the escape route costs four weeks whether or not the seam exists, because the expensive parts
— IAM, pipelines, identity federation, re-pricing — are not what the seam abstracts.

Where it did pay, and this was not the argument made for it: the seam forced every provider
detail into one directory, and that made it obvious that the *configuration* was never
abstracted at all. A codebase with `gs://` scattered through `src/` would have hidden that.
The abstraction's value here was diagnostic rather than operational — it did not make
migration cheap, it made the true cost of migration visible. That is worth something, and it
is not what the brochure claims.

---

## Task 2 — least privilege, and the experiment that failed usefully

### The three identities

| identity | may do | why each permission is needed |
|:--|:--|:--|
| **training** (`itcs355-train@`) | `storage.objects.get` on `gs://itcs355-6688010/itcs355/*`; `storage.objects.create` on `…/artifacts/*`; `aiplatform.customJobs.create`; `logging.logEntries.create` | Reads the DVC-tracked dataset; writes the model artifact and metrics back; creates its own job; writes logs. It has **no** read access to the model registry and **no** deploy permission: a training job that can deploy can put an unregistered model into production without passing the gate. |
| **serving** (`itcs355-serve@`) | `storage.objects.get` on `…/artifacts/*`; `aiplatform.endpoints.predict`; `logging.logEntries.create`; `monitoring.timeSeries.create` | Reads exactly one model artifact at startup and never writes to storage. Writes logs and custom metrics, because the drift panel and the SLO both depend on them. It cannot read the raw dataset — a serving container that can read training data turns a container escape into a data breach. |
| **CI** (`itcs355-ci@`, via Workload Identity Federation) | `artifactregistry.repositories.uploadArtifacts`; `aiplatform.pipelineJobs.create`; `aiplatform.endpoints.deploy`; `iam.serviceAccounts.actAs` on the two above | Pushes SHA-tagged images, triggers the pipeline, deploys to staging. It holds no long-lived key: GitHub mints an OIDC token and GCP exchanges it for a credential that expires in an hour. `actAs` is the one that looks harmless and is not — it is what lets CI hand work to the training and serving identities, and it is the permission that makes CI effectively as privileged as both. |

### Removing a permission, and what actually broke

The experiment: take write access away from the credential the adapter uses, and confirm
that `upload` fails.

```python
credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/devstorage.read_only"])
client = storage.Client(project="itcs355-6688010", credentials=credentials)
client.bucket("itcs355-6688010").blob("itcs355/leastprivilege/probe.txt").upload_from_string("probe")
```

**Result: the write succeeded.** Nothing broke, and that is the finding.

Diagnosing it rather than shrugging:

```
credential class : Credentials          (google.oauth2.credentials — an authorized_user)
requires_scopes  : False
scopes requested : None
scopes on the minted token:
    https://www.googleapis.com/auth/cloud-platform
    https://www.googleapis.com/auth/userinfo.email  openid  email  sqlservice.login
```

The scopes passed to `google.auth.default()` were **discarded**. Application Default
Credentials here are a user credential, and a user credential's refresh token carries the
scopes granted at `gcloud auth application-default login` — `cloud-platform`, everything.
`requires_scopes` is `False`, so google-auth does not even attempt to downscope, and the
minted access token has full project access no matter what the code asks for.

**What this teaches, and it is the point of the task.** Asking for a narrower scope is not a
security boundary. It is a request that a service-account credential will honour and a user
credential will silently ignore, and nothing in the API tells you which you have. Real
restriction on GCP comes from the IAM role bound to the principal, and the only way to test
it is to run the code as a principal that genuinely lacks the permission — a service account
with a narrow role, not a scope argument on a developer's own credential.

The practical consequence for this project: **every local run so far has executed with full
project access.** The identity table above describes the intended boundaries; it has not yet
been enforced, because enforcing it needs the three service accounts, and creating them is
part of the session that provisions managed compute. Until then the table is a design, not a
control, and calling it a control would be the exact mistake this task exists to catch.
