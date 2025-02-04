import os
from sentence_transformers import SentenceTransformer

model_parent_dir_name = 'models'
default_embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
model_env_var = 'SENTENCE_TRANSFORMERS_TEXT_EMBEDDING_MODEL'


def main():
    model_parent_dir = os.path.join(os.getcwd(), model_parent_dir_name)
    print(f"Model parent directory: {model_parent_dir}")
    # Check if the directory exists
    if not os.path.exists(model_parent_dir):
        print(f"Creating model parent directory: {model_parent_dir}")
        os.makedirs(model_parent_dir)

    # Check if model is already downloaded
    model_name = default_embedding_model
    model_dir = os.path.join(model_parent_dir, model_name)
    if os.path.exists(model_dir):
        print(f"Model {model_name} already downloaded")
        return

    # Initialize and download the model
    print(f"Downloading {model_name}...")
    model = SentenceTransformer(model_name)
    model.save(model_dir)


def get_hf_embeddings(text: str) -> list:
    model_dir = os.path.join(
        os.getcwd(), model_parent_dir_name, default_embedding_model)
    model = SentenceTransformer(model_dir)
    embeddings = model.encode(text)
    return embeddings.tolist()


if __name__ == "__main__":
    # main()
    embeddings = get_hf_embeddings(
        "I am doing hard research on the quantization techniques in embeddings and LLM weights ")
    print(embeddings)
