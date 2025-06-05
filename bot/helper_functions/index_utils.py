"""
LlamaIndex utilities for document processing and querying
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.core.indices import SummaryIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

class IndexUtils:
    """Utilities for document indexing and querying with LlamaIndex"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        
    def summarize(self, data_folder):
        """
        Generate a summary of documents in a folder
        
        Args:
            data_folder: Path to folder containing documents
            
        Returns:
            Summary response
        """
        # Initialize a document
        documents = SimpleDirectoryReader(data_folder).load_data()
        summary_index = SummaryIndex.from_documents(documents)
        
        # SummaryIndexRetriever
        retriever = summary_index.as_retriever(retriever_mode='default')
        
        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize",
            summary_template=self.config.summary_template,
        )
        
        # Assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        response = query_engine.query(
            "Generate a summary of the input context. Be as verbose as possible, "
            "while keeping the summary concise and to the point."
        )
        return response

    def simple_query(self, data_folder, query):
        """
        Perform a simple query on documents in a folder
        
        Args:
            data_folder: Path to folder containing documents
            query: Query string
            
        Returns:
            Query response
        """
        # Initialize a document
        documents = SimpleDirectoryReader(data_folder).load_data()
        vector_index = VectorStoreIndex.from_documents(documents)
        
        # Configure retriever
        retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=6,
        )
        
        # Configure response synthesizer
        response_synthesizer = get_response_synthesizer(
            text_qa_template=self.config.qa_template,
        )
        
        # Assemble query engine
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        response = query_engine.query(query)
        return response
