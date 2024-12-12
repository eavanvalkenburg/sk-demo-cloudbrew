from contextlib import asynccontextmanager
import os

import nest_asyncio
import qdrant_client
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.aio import SearchIndexClient
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.schema import BaseNode
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.extractors import BaseExtractor
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.readers.github import GithubClient, GithubRepositoryReader
from llama_index.vector_stores.azureaisearch import (
    AzureAISearchVectorStore,
    IndexManagement,
    MetadataIndexFieldType,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from tree_sitter import Language, Parser
from tree_sitter_python import language


# dotenv.load_dotenv()
nest_asyncio.apply()

lang = Language(language())
parser = Parser(lang)

CACHE_PERSIST_PATH = "data_ingestion/data/data"

search_service_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
index_name = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
search_service_api_version = "2024-07-01"
credential = AzureKeyCredential(os.getenv("AZURE_AI_SEARCH_API_KEY"))


metadata_fields = {
    "topic": ("topic", MetadataIndexFieldType.STRING),
    "subtopic": ("subtopic", MetadataIndexFieldType.STRING),
    "connector": ("connector", MetadataIndexFieldType.STRING),
}


def extract_tag(node: BaseNode) -> dict:
    filepath = node.metadata["file_path"]
    if filepath:
        # example filepath = python/samples/concepts/auto_function_calling/function_calling_with_required_type.py
        # topic = sample
        # subtopic = concepts
        # example filepath = python/semantic_kernel/connectors/memory/weaviate/weaviate_collection.py
        # topic = semantic_kernel
        # subtopic = memory
        # connector = weaviate
        topic = filepath.split("/")[1]
        subtopic = filepath.split("/")[2]
        if "semantic_kernel/connectors" in filepath:
            subtopic = filepath.split("/")[2]
            connector = filepath.split("/")[3]
            return {"topic": topic, "subtopic": subtopic, "connector": connector}
        return {"topic": topic, "subtopic": subtopic}
    return {}


class TagExtractor(BaseExtractor):
    async def aextract(self, nodes) -> list[dict]:
        return [extract_tag(node) for node in nodes]


def get_gh_client():
    return GithubClient(github_token=os.getenv("GITHUB_TOKEN"), verbose=False)


def get_gh_reader_sk(github_client, folder_filter):
    return GithubRepositoryReader(
        github_client=github_client,
        owner="microsoft",
        repo="semantic-kernel",
        use_parser=False,
        verbose=False,
        filter_directories=(
            folder_filter,
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [".py"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
    )


@asynccontextmanager
async def get_azure_store():
    # Use index client to demonstrate creating an index
    async with SearchIndexClient(
        endpoint=search_service_endpoint,
        credential=credential,
    ) as index_client:
        yield AzureAISearchVectorStore(
            search_or_index_client=index_client,
            filterable_metadata_field_keys=metadata_fields,
            index_name=index_name,
            index_management=IndexManagement.CREATE_IF_NOT_EXISTS,
            id_field_key="id",
            chunk_field_key="chunk",
            embedding_field_key="embedding",
            embedding_dimensionality=1536,
            metadata_string_field_key="metadata",
            doc_id_field_key="doc_id",
            language_analyzer="en.lucene",
            vector_algorithm_type="exhaustiveKnn",
        )


@asynccontextmanager
async def get_qdrant_store():
    client = qdrant_client.AsyncQdrantClient(
        host=os.getenv("QDRANT_HOST"),
        port=os.getenv("QDRANT_PORT"),
        grpc_port=os.getenv("QDRANT_GRPC_PORT"),
        prefer_grpc=False,
    )
    yield QdrantVectorStore(collection_name="sk", aclient=client, index_doc_id=False)
    await client.close()


def openai_embedder():
    return OpenAIEmbedding(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name=os.getenv("OPENAI_EMBEDDING_MODEL_ID"),
        dimensions=1536,
    )


def ollama_embedder():
    return OllamaEmbedding(model_name="llama3.2")


def get_sk_pipeline(cache, embedder):
    return IngestionPipeline(
        transformations=[
            CodeSplitter(
                language="python",
                parser=parser,
                chunk_lines=100,
                chunk_lines_overlap=30,
                max_chars=2000,
            ),
            TagExtractor(),
            embedder,
        ],
        cache=IngestionCache(cache=cache),
    )


async def get_nodes(reader: GithubRepositoryReader, pipeline: IngestionPipeline):
    return await pipeline.arun(
        show_progress=True, documents=await reader.aload_data(branch="main")
    )


async def get_azure_search_index(nodes):
    async with get_azure_store() as azure_ai_search_store:
        VectorStoreIndex(
            nodes=nodes,
            storage_context=StorageContext.from_defaults(
                vector_store=azure_ai_search_store
            ),
            vector_store_name="azure",
            metadata_fields=metadata_fields,
            id_field_key="id",
            chunk_field_key="chunk",
            embedding_field_key="embedding",
            doc_id_field_key="doc_id",
            show_progress=True,
            use_async=True,
        )


async def get_qdrant_index(nodes):
    async with get_qdrant_store() as qdrant_store:
        VectorStoreIndex(
            nodes=nodes,
            storage_context=StorageContext.from_defaults(vector_store=qdrant_store),
            vector_store_name="qdrant",
            metadata_fields=metadata_fields,
            id_field_key="id",
            chunk_field_key="chunk",
            embedding_field_key="embedding",
            doc_id_field_key="doc_id",
            show_progress=True,
            use_async=True,
        )


async def main():
    gh_client = get_gh_client()
    # Load data from a Github repository
    sk_reader = get_gh_reader_sk(
        gh_client, ["python/semantic_kernel", "python/samples"]
    )
    # samples_reader = get_gh_reader_sk(gh_client, [])

    if os.path.exists(CACHE_PERSIST_PATH) and os.path.getsize(CACHE_PERSIST_PATH) > 0:
        cache = SimpleKVStore.from_persist_path(CACHE_PERSIST_PATH)
    else:
        cache = SimpleKVStore()

    # az_sk_pipeline = get_sk_pipeline(cache, openai_embedder())
    qd_sk_pipeline = get_sk_pipeline(cache, ollama_embedder())
    # storage_context = get_storage_context()
    # az_sk_nodes = await get_nodes(sk_reader, az_sk_pipeline)
    qd_sk_nodes = await get_nodes(sk_reader, qd_sk_pipeline)
    # await get_azure_search_index(az_sk_nodes)
    await get_qdrant_index(qd_sk_nodes)
    # print(azure_ai_search_index.as_query_engine().query("What is a Agent?"))
    # print(qdrant_index.as_query_engine().query("What is a Agent?"))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
