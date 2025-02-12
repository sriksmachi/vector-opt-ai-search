from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
import azure_search_manager as asm
import os
import uuid
import json
from hf_embeddings import get_hf_embeddings
from py_embeddings import get_py_embeddings
from indexes import indexes_config

load_dotenv(override=True)
azure_openai_embedding_dimensions = 3072
hf_embedding_dimensions = 384
py_embedding_dimensions = 384
py_embedding_dimensions_packedbit = 3072

# Load environment variables
search_service_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_service_key = os.environ["AZURE_SEARCH_KEY"]
azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
azure_openai_embedding_deployment = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
azure_openai_embedding_dimensions = int(
    os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", 3072))
embedding_model_name = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

# Initialize the Azure Search and OpenAI clients
search_index_client = SearchIndexClient(
    search_service_endpoint, AzureKeyCredential(search_service_key))
openai_client = AzureOpenAI(
    azure_deployment=azure_openai_embedding_deployment,
    api_version=azure_openai_api_version,
    azure_endpoint=azure_openai_endpoint,
    api_key=azure_openai_key
)


def get_embeddings(text):
    """
    Get embeddings for the given text using Azure OpenAI.

    Args:
        text (str): The input text to get embeddings for.

    Returns:
        list: A list of embeddings for the input text.
    """
    response = openai_client.embeddings.create(
        input=text, model=embedding_model_name)
    return response.data[0].embedding


def create_indexes(indexes, base_index_name):
    azure_indexes = []
    for index, options in indexes.items():
        try:
            print("=" * 50)
            search_index_client.delete_index(index=base_index_name + index)
            print(f"Deleted index {base_index_name + index}")
        except Exception:
            pass
        dimensions = azure_openai_embedding_dimensions
        if index == "hf_embeddings":
            dimensions = hf_embedding_dimensions
        if index == "py_embeddings_scalar":
            dimensions = py_embedding_dimensions
        if index == "py_embeddings_binary":
            dimensions = py_embedding_dimensions_packedbit
        index = asm.create_index(
            base_index_name + index, dimensions=dimensions, **options)
        print(f"Creating index {index.name}")
        search_index_client.create_or_update_index(index)
        azure_indexes.append(index.name)
        print("=" * 50)
    print(f"Created indexes: {azure_indexes}")
    return azure_indexes


def get_chunks(text, chunk_size=20000, index_name="azure_openai"):
    documents = []

    # if {index_name}.json" exists, load the documents from the file
    vectors_folder = "vectors"

    if not os.path.exists(vectors_folder):
        os.makedirs(vectors_folder)

    if os.path.exists(f"{vectors_folder}/{index_name}.json"):
        print(f"Loading documents from {index_name}.json")
        with open(f"{vectors_folder}/{index_name}.json", "r") as f:
            return json.load(f)

    # split by chapter
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Splitting text into {len(chunks)} chunks")
    embeddings = None
    print(f"Processing chunks...")
    for i, chunk in enumerate(chunks):
        if index_name == "hf_embeddings":
            embeddings = get_hf_embeddings(chunk)
        elif index_name == "py_embeddings_scalar":
            embeddings = get_py_embeddings(chunk, quantization="scalar")
        elif index_name == "py_embeddings_binary":
            embeddings = get_py_embeddings(chunk, quantization="binary")
        else:
            embeddings = get_embeddings(chunk)
        documents.append({
            "id": str(uuid.uuid4()),
            "title": f"chunk-{i}",
            "chunk": chunk,
            "embedding": embeddings
        })

    with open(f"{vectors_folder}/{index_name}.json", "w") as f:
        print(f"Saving documents to {index_name}.json")
        json.dump(documents, f)

    return documents


def read_file(file_path):
    # read txt file
    with open(file_path, 'r', encoding="utf8") as file:
        text = file.read()
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
    return text


def ingest():
    """
    Ingests data files from the 'data' folder, processes them, and uploads the content to search indexes.

    This function performs the following steps:
    1. Creates search indexes based on the provided configuration.
    2. Lists all files in the 'data' folder.
    3. Reads the content of each file with a '.txt' extension.
    4. Splits the file content into chunks.
    5. Uploads the chunks to the appropriate search index.

    Returns:
        list: A list of created search indexes.
    """
    data_folder = 'data'
    indexes = create_indexes(indexes_config, "")
    files_list = os.listdir(data_folder)
    for file in files_list:
        file_path = os.path.join(data_folder, file)
        # Read the file content
        print(f"Chunking & Vectorizing....")
        if file.endswith('.txt'):
            file_content = read_file(file_path)
            # Upload the chunks to the search index
            for index in indexes:
                # Split the text into chunks
                index_name = index
                if "hf_embeddings" in index:
                    index_name = "hf_embeddings"
                elif "py_embeddings_scalar" in index:
                    index_name = "py_embeddings_scalar"
                elif "py_embeddings_binary" in index:
                    index_name = "py_embeddings_binary"
                else:
                    index_name = "azure_openai"
                chunks = get_chunks(
                    file_content, index_name=index_name)
                sea = SearchClient(search_service_endpoint, index_name=index,
                                   credential=AzureKeyCredential(search_service_key))
                print(f"Uploading {len(chunks)} chunks to {index}")
                sea.upload_documents(chunks)
    return indexes


if __name__ == "__main__":
    indexes = ingest()
    print(indexes)
