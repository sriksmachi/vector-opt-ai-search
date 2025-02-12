# Vector Optimization AI Search

As Generative AI usage increases, leveraging meaningful data through RAG architecture is crucial for custom scenarios. Azure AI Search, despite its vector storage support, faces performance issues due to memory demands. This session covers vector compression via quantization and re-ranking through oversampling to optimize query performance, cost and search quality.


## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/vector-opt-ai-search.git
    cd vector-opt-ai-search
    ```

2. **Install the required dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

3. **Set up environment variables:**

    Create a [.env](http://_vscodecontentref_/11) file in the root directory and add the following variables:

    ```env
    AZURE_SEARCH_ENDPOINT=<your-azure-search-endpoint>
    AZURE_SEARCH_KEY=<your-azure-search-key>
    AZURE_OPENAI_ENDPOINT=<your-azure-openai-endpoint>
    AZURE_OPENAI_KEY=<your-azure-openai-key>
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large
    AZURE_OPENAI_EMBEDDING_DIMENSIONS=3072
    AZURE_OPENAI_API_VERSION=2024-06-01
    ```

## Usage

1. **Run the data pipeline:**

    ```sh
    python data_pipeline.py
    ```

    This script will:
    - Load environment variables.
    - Initialize Azure Search and OpenAI clients.
    - Create search indexes based on the configuration in [indexes.py](http://_vscodecontentref_/12).
    - Read text files from the [data](http://_vscodecontentref_/13) folder.
    - Generate embeddings for the text using different models.
    - Upload the embeddings to the appropriate search indexes.

2. **Check the created indexes:**

    The created indexes will be printed in the console output.

## Files

- **`data_pipeline.py`**: Main script for processing text data, generating embeddings, and indexing the data in Azure Search.
- **[azure_search_manager.py](http://_vscodecontentref_/14)**: Contains functions for managing Azure Search indexes.
- **[hf_embeddings.py](http://_vscodecontentref_/15)**: Functions for generating embeddings using Hugging Face models.
- **[py_embeddings.py](http://_vscodecontentref_/16)**: Functions for generating embeddings using custom Python models.
- **[indexes.py](http://_vscodecontentref_/17)**: Configuration for the search indexes.
- **`optimized_vector_embeddings.ipynb`**: Jupyter notebook for experimenting with vector embeddings.
- **[data](http://_vscodecontentref_/18)**: Folder containing text files to be processed.
- **[vectors](http://_vscodecontentref_/19)**: Folder where generated vectors are saved.

## License

This project is licensed under the MIT License. See the LICENSE file for details.