"""Lab 5 — compile pipeline/pipeline.yaml into your provider's pipeline service.

This module is Layer 3, so provider SDKs are allowed here and nowhere else. The DAG stays
in YAML; only the translation lives here. That separation is what you argue for or against
in Lab 5 Task 4.

Implement ONE compile function, for your provider.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

SPEC = Path(__file__).resolve().parents[1] / "pipeline" / "pipeline.yaml"


def load_spec(path: Path = SPEC) -> dict[str, Any]:
    """Read the neutral DAG and expand ${VAR} references from the environment."""
    raw = path.read_text()
    for key, value in os.environ.items():
        raw = raw.replace(f"${{{key}}}", value)
    return yaml.safe_load(raw)


def compile_aws(spec: dict[str, Any]):
    """SageMaker Pipelines.

    Each step becomes a ProcessingStep or TrainingStep; `condition` becomes a
    ConditionStep. Note that SageMaker's condition steps branch rather than abort, so
    `on_failure: abort` needs a FailStep on the else branch — an asymmetry worth
    mentioning in your Lab 5 write-up.
    """
    raise NotImplementedError("TODO Lab 5: build a sagemaker.workflow.pipeline.Pipeline")


def compile_azure(spec: dict[str, Any]):
    """Azure ML Pipelines.

    Each step becomes a command component; the DAG is expressed as a @pipeline function.
    Conditions use azure.ai.ml.dsl.condition, which is closer to this YAML than the other
    two providers' equivalents.
    """
    raise NotImplementedError("TODO Lab 5: build an azure.ai.ml pipeline job")


def gate_producer(spec: dict[str, Any]) -> str | None:
    """Name the step that emits the gate decision, if any."""
    for step in spec["steps"]:
        if "gate_decision" in step.get("outputs", []):
            return step["name"]
    return None


def gated_steps(spec: dict[str, Any]) -> set[str]:
    """Every step that must sit behind the gate.

    That is the step carrying the `condition`, plus everything downstream of it. Deploying
    is downstream of registering, so a compiler that put only the conditional step inside
    the branch would deploy a model it had just refused to register — and KFP would not
    catch it, because "deploy an unregistered model" is a perfectly legal DAG.
    """
    behind_gate = {step["name"] for step in spec["steps"] if step.get("condition")}
    if not behind_gate:
        return set()

    changed = True
    while changed:
        changed = False
        for step in spec["steps"]:
            if step["name"] in behind_gate:
                continue
            if any(upstream in behind_gate for upstream in step.get("depends_on", [])):
                behind_gate.add(step["name"])
                changed = True
    return behind_gate


def _container_component_for(step: dict[str, Any]):
    """Build one KFP container component from a step in the neutral DAG.

    The component is defined by executing generated source, which deserves an explanation.
    KFP v2 takes a component's name from the decorated function's `__name__`, and setting
    it afterwards does not stick — `component_spec.name` is read at decoration time. A
    pipeline whose steps were all called "step" would be unreadable in the Vertex console
    during an incident, which is the one moment the names matter. Generating the def is the
    only way to give a component a name that came out of the YAML.

    A step declaring `gate_decision` in its outputs gets an output parameter and is handed
    `--decision-out <path>`, which is what lets a later step branch on the result.
    """
    from kfp import dsl

    identifier = step["name"].replace("-", "_")
    produces_gate_decision = "gate_decision" in step.get("outputs", [])

    namespace = {"dsl": dsl, "image": step["image"], "command": step["command"]}
    if produces_gate_decision:
        source = (
            f"@dsl.container_component\n"
            f"def {identifier}(gate_decision: dsl.OutputPath(str)):\n"
            f"    return dsl.ContainerSpec(image=image, command=command,\n"
            f"                             args=['--decision-out', gate_decision])\n"
        )
    else:
        source = (
            f"@dsl.container_component\n"
            f"def {identifier}():\n"
            f"    return dsl.ContainerSpec(image=image, command=command)\n"
        )
    # dont_inherit=True matters: without it the generated code inherits this module's
    # `from __future__ import annotations`, the OutputPath annotation arrives at KFP as the
    # string "dsl.OutputPath(str)", and the compiler rejects it as a malformed artifact
    # type. The error names artifact schemas and says nothing about futures.
    compiled_source = compile(source, "<generated component>", "exec", dont_inherit=True)
    exec(compiled_source, namespace)
    return namespace[identifier]


def compile_gcp(spec: dict[str, Any], output_path: Path | None = None) -> Path:
    """Vertex AI Pipelines.

    Kubeflow Pipelines underneath: steps become container components, conditions become
    dsl.If blocks. The compiled artifact is a file you submit, so this returns a path where
    the other two providers' compilers would return an object — a real asymmetry in the
    seam, and one of the leaks the Lab 5 write-up is about.

    The `condition` in the YAML becomes a genuine branch. Compile and look for a
    `comp-condition-*` component in the output: if it is missing, the gate is decorative.
    """
    from kfp import compiler, dsl

    components = {step["name"]: _container_component_for(step) for step in spec["steps"]}
    destination = output_path or Path("reports") / "pipeline-vertex.yaml"

    @dsl.pipeline(name=spec["name"], description="ITCS355 training pipeline")
    def pipeline_function():
        """The DAG, assembled in dependency order from pipeline/pipeline.yaml."""
        gated = gated_steps(spec)
        tasks: dict[str, Any] = {}

        def build(step: dict[str, Any]) -> None:
            """Instantiate one step and wire its upstream dependencies."""
            task = components[step["name"]]()
            for upstream in step.get("depends_on", []):
                task.after(tasks[upstream])
            tasks[step["name"]] = task

        gate_step_name = gate_producer(spec)
        for step in spec["steps"]:
            if step["name"] not in gated:
                build(step)

        if not gated:
            return

        gate_task = tasks[gate_step_name]
        with dsl.If(gate_task.outputs["gate_decision"] == "pass", name="gate-passed"):
            for step in spec["steps"]:
                if step["name"] in gated:
                    build(step)

    destination.parent.mkdir(parents=True, exist_ok=True)
    compiler.Compiler().compile(pipeline_function, str(destination))
    return destination


COMPILERS = {"aws": compile_aws, "azure": compile_azure, "gcp": compile_gcp}


def compile_for(provider: str, spec: dict[str, Any] | None = None):
    """Compile the neutral DAG for one provider."""
    spec = spec or load_spec()
    try:
        return COMPILERS[provider.lower()](spec)
    except KeyError:
        raise ValueError(f"No pipeline compiler for {provider!r}") from None
