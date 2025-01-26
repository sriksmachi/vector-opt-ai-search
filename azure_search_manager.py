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
    VectorSearchAlgorithmKind,
    HnswParameters,
    VectorSearchAlgorithmMetric,
    VectorEncodingFormat
)


def upload_documents(index, document_id, document_text):
    documents = [
        {
            "id": document_id,
            "title": document_id,
            "chunk": document_text,
            "embedding": fetch_embeddings(document_text)["embedding"]
        }
    ]
    index.upload_documents(documents)


def create_index(index_name, dimensions, use_scalar_compression=False, use_binary_compression=False, vector_type="Collection(Edm.Single)",
                 use_stored=True,
                 truncation_dimension=None):

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

    if use_scalar_compression:
        compression_name = "myCompression"
        compression_configurations = [
            ScalarQuantizationCompression(
                compression_name=compression_name, truncation_dimension=truncation_dimension)
        ]

    elif use_binary_compression:
        compression_name = "myCompression"
        compression_configurations = [
            BinaryQuantizationCompression(
                compression_name=compression_name, truncation_dimension=truncation_dimension)
        ]
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
