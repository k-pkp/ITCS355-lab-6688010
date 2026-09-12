<!-- Lab 4 Task 6. Written against docs/postmortem-template.md; the measurements it
     cites are in reports/lab4-drift.md. -->

# Post-mortem — temperature readings jumped 6 °C across the fleet

**What fired:** PSI on `temp_c` reached 0.3833 against a reference window, crossing the
0.20 alert threshold, 2026-09-07. KS on the same feature was 0.2457. Every other feature
scored exactly 0.0000 — the giveaway.

**True cause:** not drift. All six features moving together would be a fleet change; one
feature moving alone, by a constant, with the other five bit-identical, is arithmetic
applied upstream — a recalibration, a unit conversion, or a firmware update on the sensor
gateway. The distribution's shape is unchanged; it has been translated. Nothing about the
machines changed, only what the pipeline said about them.

**Retrain, roll back, or no action — and why:** **no action on the model; fix the
producer.** Retraining here is the worst available option: it would bake a broken
calibration into the next model as though it were the world, and when the upstream fix
lands the newly trained model would be wrong in the opposite direction, with the last good
model already replaced. Roll back is also wrong — the model did not change, the input did.
The correct sequence is to hold the current version, confirm with the gateway owner what
changed and when, backfill the corrected readings, and only then ask whether anything needs
retraining. What would change my mind: if the other five features had moved too, or if the
shape had changed rather than the location, that would be a real fleet change and the
answer would be retrain on recent data.

**What this would have cost if unnoticed for a week:** measured, −0.0100 ROC AUC. On 240
machines checked daily that is 1,680 predictions a week with the ranking mildly degraded —
on this positive rate, roughly one or two genuinely at-risk machines slipping below the
triage cut-off per week, and a similar number of unnecessary call-outs. At an assumed 2,000
THB per unnecessary technician visit that is a few thousand THB a week of wasted labour,
against a compute bill for the whole system of under 10 THB. The model is not the expensive
part; the decisions it drives are. The far larger risk is silent: had this been retrained
into rather than investigated, the corrupted calibration becomes permanent and the cost
runs until someone re-derives it from first principles.

**How to prevent or detect it faster:** add a data contract test asserting that
`temp_c`'s rolling weekly mean stays within 3 standard deviations of the reference mean,
and run it in the ingest step of the pipeline, before training — the same place the schema
and null-rate checks already run. That converts this from a monitoring alert that someone
must interpret a week later into a pipeline abort at the moment the bad batch arrives, and
it is the branch of the Lab 4 decision tree that matters: a model trained on data that
failed its own contract is one nobody should trust.
