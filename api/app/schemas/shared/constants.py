import os

from dotenv import load_dotenv

load_dotenv()

LANGUAGE = "cz"

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", LANGUAGE)
