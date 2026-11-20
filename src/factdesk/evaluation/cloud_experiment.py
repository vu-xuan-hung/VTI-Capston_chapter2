"""Run the same cases and evaluators on LangSmith: langsmith.evaluate.

Needs LANGSMITH_API_KEY. langsmith is imported inside the function, so the
local path (experiment.py) never needs an account.
"""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

from    collections.abc                 import  Callable
from    typing                          import  Any

from    factdesk.evaluation.dataset     import  Example
from    factdesk.observability.cloud    import  require_cloud


#==========================================================================
# Cloud experiment
#==========================================================================
def run_cloud_experiment(
    name: str,
    target: Callable[[dict[str, Any]], dict[str, Any]],
    dataset: list[Example],
    evaluators: list[Callable[..., Any]],
    *,
    dataset_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """
    Create the LangSmith dataset once, then run langsmith.evaluate with the
    same target and evaluator functions as the local experiment. A dataset
    that exists with another number of cases stops the run: rename it when
    the cases change, so results stay comparable.

    Input:
        . name          : experiment prefix, e.g. "v2"
        . target        : inputs dict -> outputs dict
        . dataset       : the cases (uploaded only when the dataset is new)
        . evaluators    : the same functions as run_experiment uses
        . dataset_name  : the LangSmith dataset, e.g. config.tracing.dataset
        . metadata      : experiment metadata, e.g. {"backend": "local"}
    Output:
        . experiment    : the LangSmith experiment name
    """

    #----------------------------------------------------------------------
    # Fail before any network call when there is no key
    #----------------------------------------------------------------------
    require_cloud()
    from    langsmith                       import  Client, evaluate

    #----------------------------------------------------------------------
    # Create the dataset once
    #----------------------------------------------------------------------
    client = Client()
    if not client.has_dataset(dataset_name=dataset_name):
        client.create_dataset(dataset_name, description=f"{len(dataset)} cases")
        client.create_examples(
            dataset_name    = dataset_name,
            examples        = [
                {
                    "inputs": example.inputs,
                    "outputs": example.outputs,
                    "metadata": {"case_id": example.id, "split": example.split},
                }
                for example in dataset
            ],
        )
    stored = sum(1 for _ in client.list_examples(dataset_name=dataset_name))
    if stored != len(dataset):
        raise ValueError(
            f"LangSmith dataset {dataset_name!r} has {stored} cases, this run has {len(dataset)}: "
            "use a new dataset name for a changed case set"
        )

    #----------------------------------------------------------------------
    # Evaluate
    #----------------------------------------------------------------------
    results = evaluate(
        target,
        data                = dataset_name,
        evaluators          = evaluators,
        experiment_prefix   = name,
        metadata            = metadata or {},
    )
    return results.experiment_name
