from typing import List
import requests
import numpy as np


def fetch_embeddings(query_text: str) -> dict:
    # Define the URL and the data payload
    endpoint = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "mxbai-embed-large:latest",
        "prompt": query_text
    }

    # Send the POST request
    response = requests.post(url=endpoint, json=payload)

    # Print the response
    return response.json()


def scalar_quantization(input_vector: List[float], quantization_type=np.uint8) -> List[int]:
    # Calculate the min and max for each dimension
    min_vals = np.min(input_vector, axis=0)
    max_vals = np.max(input_vector, axis=0)

    print(min_vals, max_vals)

    # Calculate scaling factor and zero point for each dimension
    scaling_factors = (max_vals - min_vals) / 255.0
    zero_points = -min_vals / scaling_factors

    # Quantize the embeddings
    quantized_embeddings = np.round(
        (input_vector - min_vals) / scaling_factors).astype(quantization_type)

    return quantized_embeddings.tolist()


def binary_quantization(input_vector: List[float]) -> List[int]:
    # Convert embeddings to -1 or 1 based on their sign
    binary_embeddings = np.where(np.array(input_vector) >= 0, 1, -1)
    return binary_embeddings.tolist()


vector_response = fetch_embeddings(
    query_text="I am doing hard research on the quantization techniques in embeddings and LLM weights "
               "and its impact on hallucination")

embedding = vector_response['embedding']
print('Original Vector:')
print(vector_response['embedding'])

scalar_quantized_vector = scalar_quantization(
    input_vector=embedding, quantization_type=np.uint16)
print('\nScalar Quantized Vector:')
print(scalar_quantized_vector)

binary_quantized_vector = binary_quantization(input_vector=embedding)
print('\nBinary Quantized Vector:')
print(binary_quantized_vector)
