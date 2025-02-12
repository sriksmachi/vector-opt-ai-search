
indexes_config = {
    "baseline": {},
    "binary-compression": {
        "use_binary_compression": True
    },
    "no-stored": {
        "use_stored": False
    },
    "all-options-with-scalar": {
        "use_scalar_compression": True,
        "use_float16": True,
        "use_stored": False,
    },
    "all-options-with-binary": {
        "use_binary_compression": True,
        "use_float16": True,
        "use_stored": False,
    },
    "scalar-oversampling-reranking": {
        "use_scalar_compression": True,
        "use_oversampling_reranking": True,
        "use_float16": True,
        "use_stored": False,
    },
    "scalar-truncation": {
        "use_truncation": True,
        "use_scalar_compression": True,
        "use_float16": True,
        "use_stored": False,
    },
    "hf_embeddings": {
        "use_hf_embeddings": True,
    },
    "py_embeddings_scalar": {
        "use_py_embeddings": True,
        "use_sbyte": True,
    },
    "py_embeddings_binary": {
        "use_py_embeddings": True,
        "use_byte": True,
    },
}
