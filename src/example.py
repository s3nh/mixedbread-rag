# Initialize with local model path
LOCAL_MODEL_PATH = "/path/to/your/local/mixedbread-ai/mxbai-embed-large-v1"
retriever = DocumentRetriever(
    collection_name="local_mixedbread_documents",
    local_model_path=LOCAL_MODEL_PATH
)

# Example usage
files_to_process = [
    "/path/to/document1.pdf",
    "/path/to/document2.md"
]

# Add custom metadata
metadata = [
    {
        "source": "document1.pdf",
        "category": "technical",
        "date": "2025-01-03",
        "added_by": "s3nh"
    },
    {
        "source": "document2.md",
        "category": "documentation",
        "date": "2025-01-03",
        "added_by": "s3nh"
    }
]

# Add documents
num_added = retriever.add_documents(files_to_process, metadata)
print(f"Added {num_added} documents")

# Search example
results = retriever.search(
    query="technical specifications",
    n_results=2
)

# Print results
for i, (doc, meta, distance) in enumerate(zip(
    results["documents"],
    results["metadatas"],
    results["distances"]
)):
    print(f"\nResult {i+1}:")
    print(f"Source: {meta['source']}")
    print(f"Added by: {meta['added_by']}")
    print(f"Similarity Score: {1 - distance:.4f}")
    print(f"Preview: {doc[:200]}...")
