from elasticsearch import Elasticsearch

def extract_es_features_optimized(es, index_name, field="content", doc_batch_size=100, analysis_batch_size=50):
    """
    Optimized feature extraction with batched document retrieval
    and batched analysis
    """
    vocab = set()
    
    try:
        # Batch document retrieval using scroll
        response = es.search(
            index=index_name,
            body={"query": {"match_all": {}}},
            scroll='2m',
            size=doc_batch_size
        )
        
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        

        while len(hits) > 0:
            # Extract texts from current batch of documents
            texts = [hit["_source"]["content"] for hit in hits if hit["_source"].get("content")]
            
            # Process texts in analysis batches
            for i in range(0, len(texts), analysis_batch_size):
                batch = texts[i:i + analysis_batch_size]
                # Filter out empty texts
                batch = [text for text in batch if text and text.strip()]
                
                if not batch:
                    continue
                    
                combined_text = " ".join(batch)
                
                try:
                    tokens = es.indices.analyze(
                        index=index_name,
                        body={
                            "field": field,
                            "text": combined_text
                        }
                    )["tokens"]
                    vocab.update([t["token"] for t in tokens])
                except Exception as e:
                    print(f"Error analyzing batch: {e}")
                    continue
            
            # Get next batch of documents
            response = es.scroll(scroll_id=scroll_id, scroll='2m')
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
        
        # Clean up scroll context
        es.clear_scroll(scroll_id=scroll_id)
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return []
    
    return sorted(vocab)





from elasticsearch import Elasticsearch, helpers
import sys
import os
from pathlib import Path
import time

def extract_features_from_documents(documents, index_name="bm25-demo", es_host="http://localhost:9200", field="content"):
    """
    Extract features from documents using Elasticsearch and RankingSHAP.
    
    Args:
        documents (list): List of dictionaries with document content
                         Format: [{"content": "text1"}, {"content": "text2"}, ...]
        index_name (str): Name of the Elasticsearch index to create
        es_host (str): Elasticsearch host URL
        field (str): Field name to analyze for features
    
    Returns:
        features: Extracted features from RankingSHAP, or None if error occurs
    """
    
    # 1. Connect to ES with error handling
    try:
        es = Elasticsearch(es_host)
        if not es.ping():
            raise ConnectionError("Cannot connect to Elasticsearch")
        print("Successfully connected to Elasticsearch")
    except Exception as e:
        print(f"ES connection failed: {e}")
        return None
    
    # 2. Delete index if it exists (for repeatable execution)
    try:
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            print(f"Deleted existing index '{index_name}'")
    except Exception as e:
        print(f"Error deleting index: {e}")
        return None
    
    # 3. Create index with BM25 (standard analyzer by default)
    mapping = {
        "mappings": {
            "properties": {
                field: {
                    "type": "text",  # Uses BM25 and standard analyzer by default
                }
            }
        }
    }
    
    try:
        es.indices.create(index=index_name, **mapping)
        print(f"Created index '{index_name}'.")
    except Exception as e:
        print(f"Error creating index: {e}")
        return None
    
    # 4. Index the provided documents
    if not documents:
        print("No documents provided")
        return None
    
    actions = [{"_index": index_name, "_source": doc} for doc in documents]
    
    try:
        response = helpers.bulk(es, actions)
        print(f"Successfully indexed {len(documents)} documents")
    except helpers.BulkIndexError as e:
        print(f"Bulk indexing errors: {e.errors}")
        return None
    except Exception as e:
        print(f"Unexpected error during bulk indexing: {e}")
        return None

    # # Refresh index to make documents searchable immediately
    try:
        es.indices.refresh(index=index_name)
        print("Index refreshed")
    except Exception as e:
        print(f"Error refreshing index: {e}")
    
    # 5. Import custom module and extract features
    try:
        # Get the current script's directory
        current_dir = Path.cwd()
        rankingshap_path = current_dir / "RankingSHAP" / "RankingShap"
        
        # Add to path if it exists
        if rankingshap_path.exists():
            sys.path.insert(0, str(rankingshap_path))
        else:
            # Fallback to your original path (make it configurable)
            fallback_path = "/Users/keqiaoli/Desktop/RankingSHAP/RankingShap/"
            if os.path.exists(fallback_path):
                sys.path.append(fallback_path)
            else:
                raise ImportError(f"RankingSHAP module not found at expected locations")
        
        from rankingSHAP_test_class_cutomized import extract_es_features_optimized
        
        # Extract features using the optimized method
        features = extract_es_features_optimized(es, index_name, field=field)
        print("Extracted features:")
        print(features)
        return features
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please check the path to your RankingSHAP module")
        return None
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

