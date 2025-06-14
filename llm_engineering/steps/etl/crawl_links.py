from typing import List, Optional
from pathlib import Path
from llm_engineering.application.crawlers.local_loader import LocalFileLoader
from llm_engineering.domain.documents import UserDocument
from llm_engineering.domain.cleaned_documents import LocalFileDocument
from typing_extensions import Annotated
from zenml import get_step_context, step
from llm_engineering.settings import settings
from llm_engineering.infrastructure.db.mongo import connection

@step
def crawl_links(
    links: Optional[List[str]] = None, 
    user: Optional[UserDocument] = None
) -> Annotated[List[str], "crawled_links"]:

    # Get and validate folder path
    folder_path = Path(settings.LOCAL_DATA_FOLDER).expanduser()

    if user is None:
        from llm_engineering.domain.documents import UserDocument
        user = UserDocument(
            first_name="System",
            last_name="User",
            
        )


    if not folder_path.exists():
        raise FileNotFoundError(f"Data folder not found: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    try:
        # Load documents (returns List[LocalFileDocument])
        loader = LocalFileLoader(str(folder_path))
        documents = loader.load_documents()

        if not documents:
            return ["No documents found in folder - nothing to process"]
        
        # Prepare documents for insertion
        documents_to_insert = [doc.model_dump() for doc in documents]
        
        # Save to MongoDB - using explicit collection name
        db = connection[settings.DATABASE_NAME]
        collection = db["localfiles"]
        result = collection.insert_many([doc.model_dump() for doc in documents])
        
        # Log results
        step_context = get_step_context()
        metadata = {
            "loaded_count": len(documents),
            "sample_file": documents[0].file_path if documents else None,
            "folder": str(folder_path.absolute())
        }
        step_context.add_output_metadata(output_name="crawled_links", metadata=metadata)
        
        return [f"Loaded {len(documents)} documents from {folder_path}"]
    
    except Exception as e:
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="crawled_links",
            metadata={
                "error": str(e),
                "failed_path": str(folder_path),
                "document_type": "LocalFileDocument"  # Added clarity
            }
        )
        raise
