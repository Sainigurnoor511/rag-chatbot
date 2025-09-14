import os
import asyncio
import shutil
from datetime import datetime, timedelta
from app.config.logger import logger

# Constants
DELETE_AFTER_MINUTES = 30  # Session expiry time in minutes

# Store session files with timestamp
SESSION_FILES = {}  # {session_id: {"file_path": str, "embedding_path": str, "timestamp": datetime}}


# Register Uploaded File and Embedding
def register_file(session_id: str, file_path: str, embedding_path: str):
    """
    Register uploaded file and embedding with session ID and timestamp.
    """
    try:
        SESSION_FILES[session_id] = {
            "file_path": file_path,
            "embedding_path": embedding_path,
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error while registering file: {str(e)}")
        raise 


# Delete Files and Embeddings
def delete_files(file_path: str, embedding_path: str):
    """
    Delete the uploaded file and corresponding embeddings.
    """
    try:
        # Delete uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted uploaded file: {file_path}")
        else:
            logger.warning(f"File not found for deletion: {file_path}")
    
    except Exception as e:
        logger.error(f"Error while deleting file: {str(e)}")
        # Don't raise here, continue with embedding cleanup

    try:
        # Delete embedding folder
        if os.path.exists(embedding_path):
            shutil.rmtree(embedding_path)
            logger.info(f"Deleted embeddings folder: {embedding_path}")
        else:
            logger.warning(f"Embedding folder not found for deletion: {embedding_path}")

    except Exception as e:
        logger.error(f"Error while deleting embeddings folder: {str(e)}")


# Background Task for Cleanup
async def cleanup_expired_files():
    """
    Periodically delete expired files and embeddings.
    """
    while True:
        try:
            now = datetime.now()
            expired_sessions = []

            for session_id, details in list(SESSION_FILES.items()):
                elapsed_time = now - details["timestamp"]

                # Check for expired sessions
                if elapsed_time > timedelta(minutes=DELETE_AFTER_MINUTES):
                    delete_files(details["file_path"], details["embedding_path"])
                    expired_sessions.append(session_id)

            # Remove expired sessions from tracking
            for session_id in expired_sessions:
                del SESSION_FILES[session_id]
                
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

        # Sleep before next cleanup check
        await asyncio.sleep(60)  # Run every minute
