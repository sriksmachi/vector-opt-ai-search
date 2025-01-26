from tqdm import tqdm
from openai import AzureOpenAI
from pypdf import PdfReader
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.indexes import SearchIndexClient
from dotenv import load_dotenv
import azure_search_manager as asm
import os
import uuid
import json

load_dotenv(override=True)

indexes_config = {
    "baseline": {}
}

base_index_name = "data-"
azure_openai_embedding_dimensions = 3072

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
    response = openai_client.embeddings.create(
        input=text, model=embedding_model_name, dimensions=azure_openai_embedding_dimensions)
    return response.data[0].embedding


def create_indexes(indexes, base_index_name):
    azure_indexes = []
    for index, options in indexes.items():
        index = asm.create_index(
            base_index_name + index, dimensions=azure_openai_embedding_dimensions, **options)
        print(f"Creating index {index.name}")
        search_index_client.create_or_update_index(index)
        azure_indexes.append(index.name)
    return azure_indexes


def get_chunks(text, chunk_size=20000, index_name="baseline"):
    documents = []

    # if {index_name}.json" exists, load the documents from the file
    if os.path.exists(f"{index_name}.json"):
        print(f"Loading documents from {index_name}.json")
        with open(f"{index_name}.json", "r") as f:
            return json.load(f)

    # split by chapter
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    print(f"Splitting text into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i}")
        documents.append({
            "id": str(uuid.uuid4()),
            "title": f"chunk-{i}",
            "chunk": chunk,
            "embedding": get_embeddings(chunk)
        })

    with open(f"{index_name}.json", "w") as f:
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
    data_folder = 'data'
    indexes = create_indexes(indexes_config, base_index_name)
    files_list = os.listdir(data_folder)
    for file in files_list:
        file_path = os.path.join(data_folder, file)
        # Read the file content
        if file.endswith('.txt'):
            file_content = read_file(file_path)
            # Split the text into chunks
            chunks = get_chunks(file_content)
            # Upload the chunks to the search index
            for index in tqdm(indexes):
                sea = SearchClient(search_service_endpoint, index_name=index,
                                   credential=AzureKeyCredential(search_service_key))
                print(f"Uploading {len(chunks)} chunks to {index}")
                sea.upload_documents(chunks)
        break
    return indexes


if __name__ == "__main__":
    indexes = ingest()
    print(indexes)
