from typing import List
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex,
    SearchField,
    ScalarQuantizationCompression,
    BinaryQuantizationCompression,
    VectorSearchCompression,
    ScalarQuantizationParameters,
)


# Create an index with the given parameters
def create_index(index_name, dimensions,
                 use_scalar_compression=False,
                 use_binary_compression=False,
                 use_float16=False,
                 use_stored=True,
                 use_truncation=False,
                 use_oversampling_reranking=False,
                 use_hf_embeddings=False,
                 use_py_embeddings=False
                 ):

    if use_float16:
        vector_type = "Collection(Edm.Half)"
    else:
        vector_type = "Collection(Edm.Single)"

    # Vector fields that aren't stored can never be returned in the response
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String,
                    key=True, sortable=True, filterable=True),
        SearchField(name="title", type=SearchFieldDataType.String),
        SearchField(name="chunk", type=SearchFieldDataType.String),
        SearchField(name="embedding", type=vector_type, searchable=True, stored=use_stored,
                    vector_search_dimensions=dimensions, vector_search_profile_name="myHnswProfile")
    ]

    compression_configurations: List[VectorSearchCompression] = []

    if use_scalar_compression and not use_oversampling_reranking:
        compression_name = "scalar_compression"
        compression_configurations = [
            ScalarQuantizationCompression(
                compression_name=compression_name,
                kind="scalarQuantization",
                parameters={"quantizedDataType": "int8"}
            )
        ]
    elif use_truncation:
        compression_params = {
            "compression_name": f"truncation-compression",
        }
        compression_name = "truncation"
        compression_configurations = [
            ScalarQuantizationCompression(
                parameters=ScalarQuantizationParameters(
                    quantized_data_type="int8"
                ),
                **compression_params)
        ]
    elif use_binary_compression and not use_oversampling_reranking:
        compression_name = "binary_compression"
        compression_configurations = [
            BinaryQuantizationCompression(
                compression_name=compression_name,
                kind="binaryQuantization"
            )
        ]
    elif use_scalar_compression and use_oversampling_reranking:
        compression_name = "oversampling_reranking"
        compression_configurations = [
            ScalarQuantizationCompression(
                compression_name=compression_name,
                default_oversampling=10,
                rerank_with_original_vectors=True
            )
        ]
    elif use_binary_compression and use_oversampling_reranking:
        compression_name = "oversampling_reranking"
        compression_configurations = [
            BinaryQuantizationCompression(
                compression_name=compression_name,
                default_oversampling=10,
                rerank_with_original_vectors=True
            )
        ]
    elif use_hf_embeddings or use_py_embeddings:
        compression_name = None
        compression_configurations = []
    else:
        compression_name = None
        compression_configurations = []
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="myHnsw")
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile", algorithm_configuration_name="myHnsw", compression_name=compression_name)
        ],
        compressions=compression_configurations
    )
    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=SemanticPrioritizedFields(
            title_field=SemanticField(field_name="title"),
            content_fields=[SemanticField(field_name="chunk")]
        )
    )
    semantic_search = SemanticSearch(configurations=[semantic_config])
    return SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic_search)
