import os
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv(override=True)
search_service_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
search_service_key = os.environ["AZURE_SEARCH_KEY"]
search_index_client = SearchIndexClient(
    search_service_endpoint, AzureKeyCredential(search_service_key))
indexes = search_index_client.list_indexes()
for index in indexes:
    search_index_client.delete_index(index.name)
print("All indexes deleted")
