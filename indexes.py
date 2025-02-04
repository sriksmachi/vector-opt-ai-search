
indexes_config = {
    "baseline": {},
    "scalar-compression": {
        "use_scalar_compression": True
    },
    "binary-compression": {
        "use_binary_compression": True
    },
    "narrow": {
        "use_float16": True
    },
    "no-stored": {
        "use_stored": False
    },
    "scalar-compresssion": {
        "use_scalar_compression": True,
    },
    "binary-compression": {
        "use_binary_compression": True,
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
        "use_hf_embeddings": True
    },
    "py_embeddings_scalar": {
        "use_py_embeddings": True,
    },
    "py_embeddings_binary": {
        "use_py_embeddings": True,
    },
}
