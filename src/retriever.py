import chromadb
from chromadb.utils import embedding_functions
import markitdown
from typing import List, Dict
from datetime import datetime

class LocalMixedBreadEmbedding(embedding_functions.HuggingFaceEmbeddingFunction):
    def __init__(self, local_model_path: str):
        """
        Initialize embedding function with locally stored model
        Args:
            local_model_path: Path to your local model directory
        """
        super().__init__(
            model_name=local_model_path,
            encode_kwargs={'normalize_embeddings': True},
            trust_remote_code=True,
            model_kwargs={'local_files_only': True}
        )

class DocumentRetriever:
    def __init__(self, collection_name: str, local_model_path: str):
        """
        Initialize retriever with local model path
        Args:
            collection_name: Name for the ChromaDB collection
            local_model_path: Path to your local mixedbread model
        """
        self.chroma_client = chromadb.Client()
        
        # Initialize embedding function with local model
        self.embedding_function = LocalMixedBreadEmbedding(
            local_model_path=local_model_path
        )
        
        self.collection = self.chroma_client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def preprocess_document(self, file_path: str) -> str:
        try:
            processed_content = markitdown.process(file_path)
            return processed_content
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return None

    def add_documents(self, file_paths: List[str], metadata: List[Dict] = None):
        documents = []
        ids = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for i, file_path in enumerate(file_paths):
            content = self.preprocess_document(file_path)
            if content:
                documents.append(content)
                ids.append(f"doc_{i}_{datetime.now().timestamp()}")
        
        if metadata is None:
            metadata = [{
                "source": file_path,
                "added_date": current_time,
                "added_by": "s3nh"  # Using your login
            } for file_path in file_paths]
        
        self.collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
        return len(documents)

    def search(self, query: str, n_results: int = 3) -> Dict:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0]
        }
    
    def search_with_filters(self, query: str, filter_dict: Dict, n_results: int = 3) -> Dict:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0]
        }

    def get_collection_stats(self) -> Dict:
        return {
            "count": self.collection.count()
        }
