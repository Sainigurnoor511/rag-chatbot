import redis
import dill as pickle
import logging
from ..config.settings import settings

# Initialize Redis client using FastAPI settings
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    db=0,
    decode_responses=False  # To handle binary data serialization
)

def save_session_to_redis(session_id: str, data: dict, expiry_seconds: int = 1200) -> bool:
    """
    Save the session history and chatbot state to Redis for 20 mins.
    """
    try:
        serialized_data = pickle.dumps(data)
        redis_client.setex(session_id, expiry_seconds, serialized_data)
        logging.info(f"Session {session_id} saved to Redis successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to save session to Redis: {e}")
        return False


def load_session_from_redis(session_id: str) -> dict:
    """
    Load the session history and chatbot state from Redis.
    """
    try:
        serialized_data = redis_client.get(session_id)
        if serialized_data:
            session_data = pickle.loads(serialized_data)
            logging.info(f"Session {session_id} loaded from Redis successfully")
            return session_data
        return None
    except Exception as e:
        logging.error(f"Failed to load session from Redis: {e}")
        return None


def delete_session_from_redis(session_id: str) -> bool:
    """
    Delete a session from Redis.
    """
    try:
        redis_client.delete(session_id)
        logging.info(f"Session {session_id} deleted from Redis")
        return True
    except Exception as e:
        logging.error(f"Failed to delete session from Redis: {e}")
        return False


def document_to_dict(doc) -> dict:
    """
    Convert a LangChain Document object to a dictionary.
    """
    if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
        return {
            'metadata': doc.metadata,
            'page_content': doc.page_content
        }
    else:
        return {}
