import json
import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    OpenAIChatCompletion,
    OpenAITextEmbedding,
)
from semantic_kernel.connectors.ai.ollama import (
    OllamaChatCompletion,
    OllamaTextEmbedding,
)
from semantic_kernel.functions import KernelParameterMetadata
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchCollection
from semantic_kernel.connectors.memory.qdrant import QdrantCollection
from semantic_kernel.data import (
    VectorStoreTextSearch,
    VectorSearchOptions,
    VectorSearchFilter,
)
from data_ingestion.datamodel import SKDataModel, SKQdrantDataModel
from online_state_service_selector import OnlineStateServiceSelector
import logging
# from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def setup():
    # load_dotenv()
    for key, value in os.environ.items():
        logger.info(f"{key}: {value}")

    remote_service_id = "online"
    local_service_id = "offline"
    kernel = Kernel(ai_service_selector=OnlineStateServiceSelector())
    kernel.add_service(OpenAIChatCompletion(service_id=remote_service_id))
    online_embedder = OpenAITextEmbedding(service_id=f"{remote_service_id}-embedding")
    kernel.add_service(online_embedder)
    kernel.add_service(
        OllamaChatCompletion(service_id=local_service_id, ai_model_id="llama3.2")
    )
    offline_embedder = OllamaTextEmbedding(
        service_id=f"{local_service_id}-embedding", ai_model_id="llama3.2"
    )
    kernel.add_service(offline_embedder)

    kernel.add_plugin(
        plugin_name="chat",
        parent_directory="/Users/edvan/Work/sk/sk-demo-cloudbrew/plugins",
    )

    azure_ai = AzureAISearchCollection(data_model_type=SKDataModel)
    qdrant = QdrantCollection(
        data_model_type=SKQdrantDataModel, collection_name="sk", named_vectors=False
    )

    azure_ai_search = VectorStoreTextSearch.from_vectorized_search(
        azure_ai,
        online_embedder,
        string_mapper=lambda x: x.chunk,
    )

    def qdrant_node_content_mapper(node):
        content = json.loads(node.node_content)
        return content.get("text", "")

    qdrant_search = VectorStoreTextSearch.from_vectorized_search(
        qdrant, offline_embedder, string_mapper=qdrant_node_content_mapper
    )

    kernel.add_functions(
        plugin_name="online_search",
        functions=[
            azure_ai_search.create_search(
                function_name="code_sample_search",
                description="A search function for samples of Semantic Kernel in python. Use this to find examples.",
                options=VectorSearchOptions(
                    filter=VectorSearchFilter.equal_to("topic", "samples"),
                    vector_field_name="embedding",
                ),
            ),
            azure_ai_search.create_search(
                function_name="code_search",
                description="Get details about the way things are called or implemented in the actual Semantic Kernel codebase.",
                options=VectorSearchOptions(
                    filter=VectorSearchFilter.equal_to("topic", "semantic_kernel"),
                    vector_field_name="embedding",
                ),
            ),
        ],
    )

    kernel.add_functions(
        plugin_name="offline_search",
        functions=[
            qdrant_search.create_search(
                function_name="code_sample_search",
                description="A search function for samples of Semantic Kernel in python. Use this to find examples.",
                options=VectorSearchOptions(
                    filter=VectorSearchFilter.equal_to("topic", "samples"),
                    vector_field_name="embedding",
                ),
                parameters=[
                    KernelParameterMetadata(
                        name="query",
                        description="The search term to use for the search",
                        type_object=str,
                        is_required=True,
                    )
                ],
            ),
            qdrant_search.create_search(
                function_name="code_search",
                description="Get details about the way things are called or implemented in the actual Semantic Kernel codebase.",
                options=VectorSearchOptions(
                    filter=VectorSearchFilter.equal_to("topic", "semantic_kernel"),
                    vector_field_name="embedding",
                ),
                parameters=[
                    KernelParameterMetadata(
                        name="query",
                        description="The search term to use for the search",
                        type_object=str,
                        is_required=True,
                    )
                ],
            ),
        ],
    )

    @kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
    async def auto_function_invocation_filter(
        context: AutoFunctionInvocationContext, next
    ):
        """A filter that will be called for each function call in the response."""
        print("\nAuto function invocation filter")
        print(f"Function: {context.function.name}")
        print(f"Calling function: {context.function.name}")
        print(f"   with arguments: {context.arguments}")
        # if we don't call next, it will skip this function, and go to the next one
        await next(context)
        print(f"Function: {context.function.name} completed")
        print(f"    with results: {context.function_result}")

    return kernel


if __name__ == "__main__":
    kernel = setup()
