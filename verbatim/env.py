import logging
import os

from dotenv import load_dotenv

LOG = logging.getLogger(__name__)


def load_env_file(env_path: str = ".env") -> bool:
    """
    Load environment variables from a .env file using python-dotenv.

    Parameters:
    - env_path (str): The path to the .env file. Default is '.env'.

    Returns:
    - bool: True if the file was successfully loaded, False otherwise.
    """
    if not os.path.isfile(env_path):
        LOG.info(f"No '{env_path}' file provided.")
        return False

    try:
        # load_dotenv returns True if the file was found and loaded, else False.
        if load_dotenv(dotenv_path=env_path, override=True):
            LOG.info(f"Environment variables from '{env_path}' loaded successfully.")
            return True
        else:
            LOG.error(f"Failed to load environment variables from '{env_path}'.")
            return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        LOG.exception(f"Error while loading '{env_path}': {e}")
        return False
