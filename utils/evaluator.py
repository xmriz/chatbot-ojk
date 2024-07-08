from llama_index.core.evaluation import RetrieverEvaluator
import pandas as pd

from llama_index.core import Response
from llama_index.core.evaluation import EvaluationResult

from typing import List

# =============================================================================
# RETRIEVER EVALUATOR


def get_retriever_evaluator(metrics, retriever):
    """Get retriever evaluator."""
    return RetrieverEvaluator.from_metric_names(metrics, retriever=retriever)


async def get_retrieval_eval_results(retriever_evaluator, qa_dataset):
    """Get evaluation results."""
    return await retriever_evaluator.aevaluate_dataset(qa_dataset)


async def get_retrieval_eval_df(name, metrics, retriever, qa_dataset):
    """Display results from evaluate."""

    retriever_evaluator = get_retriever_evaluator(metrics, retriever)

    eval_results = await get_retrieval_eval_results(retriever_evaluator, qa_dataset)

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    columns = {
        "retrievers": [name],
        **{k: [full_df[k].mean()] for k in metrics},
    }

    metric_df = pd.DataFrame(columns)

    return metric_df


# =============================================================================
# RESPONSE EVALUATOR

# define get response evaluator
def get_response_eval_df(
    query: str, response: Response, eval_result: EvaluationResult
) -> pd.DataFrame:
    # Handle empty source_nodes
    if not response.source_nodes:
        source_text = "No source available"
    else:
        source_text = response.source_nodes[0].node.text
        if len(source_text) > 1000:
            source_text = source_text[:1000] + "..."

    eval_df = pd.DataFrame(
        {
            "Query": [query],
            "Response": [str(response)],
            "Source": [source_text],
            "Evaluation Result": ["Pass" if eval_result.passing else "Fail"],
            "Reasoning": [eval_result.feedback],
        }
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    return eval_df

# =============================================================================


def get_response_eval_source_result(evaluator, query, response):

    eval_source_result_full = [
        evaluator.evaluate(
            query=query,
            response=response.response,
            contexts=[source_node.get_content()],
        )
        for source_node in response.source_nodes
    ]

    eval_source_result = [
        "Pass" if result.passing else "Fail" for result in eval_source_result_full
    ]

    return eval_source_result


# define get relevancy eval sources
def get_response_eval_sources_df(
    query: str, response: Response, evaluator
) -> None:

    eval_result = get_response_eval_source_result(
        evaluator=evaluator, query=query, response=response)

    sources = [s.node.get_text() for s in response.source_nodes]

    metadatas = [s.node.metadata for s in response.source_nodes]

    eval_df = pd.DataFrame(
        {
            "Source": sources,
            "Metadata": metadatas,
            "Eval Result": eval_result,
        },
    )
    eval_df.style.set_caption(query)
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Source"]
    )

    return (eval_df)


# =============================================================================
# BATCH RESPONSE EVALUATOR


def get_batch_eval_queries(qa_dataset, num_queries):
    queries = list(qa_dataset.queries.values())
    if num_queries > len(queries):
        num_queries = len(queries)
    queries = queries[:num_queries]
    return queries


async def get_batch_eval_results(runner, qa_dataset, query_engine, num_queries=10):
    batch_eval_queries = get_batch_eval_queries(qa_dataset, num_queries)
    eval_results = await runner.aevaluate_queries(
        query_engine=query_engine,
        queries=batch_eval_queries,
    )
    return eval_results


def get_batch_eval(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score


def get_batch_eval_df(eval_results):
    eval_results = {
        key: get_batch_eval(key, eval_results)
        for key in eval_results.keys()
    }
    eval_df = pd.DataFrame(eval_results, index=[0])
    return eval_df
