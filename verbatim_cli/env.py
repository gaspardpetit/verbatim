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
        LOG.info("No '%s' file provided.", env_path)
        return False

    try:
        # load_dotenv returns True if the file was found and loaded, else False.
        if load_dotenv(dotenv_path=env_path, override=True):
            LOG.info("Environment variables from '%s' loaded successfully.", env_path)
            return True
        LOG.error("Failed to load environment variables from '%s'.", env_path)
        return False
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOG.exception("Error while loading '%s': %s", env_path, exc)
        return False
