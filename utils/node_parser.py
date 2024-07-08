from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.ingestion import IngestionPipeline


def parse_nodes(documents: list, llm) -> list:
    node_parser = SentenceSplitter(chunk_size=1024)
    extractors = [
        QuestionsAnsweredExtractor(questions=5, llm=llm),
        SummaryExtractor(summaries=['prev', 'self'], llm=llm),
    ]

    transformations = [node_parser] + extractors
    pipeline = IngestionPipeline(transformations=transformations)

    nodes_all = pipeline.run(documents=documents)
    return nodes_all
