from enum import Enum
from typing import List
import logging

from pydantic import BaseModel

from ai_client import create_ai_client, set_pricing_file
from decouple import config

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')


# Create a client for any provider
client = create_ai_client('openai', api_key="secret-key",
                          base_url="http://localhost:8000/v1")


response1 = client.prompt(
    'command-a-03-2025',
    "Tell me a joke.",
)

print(response1.text)
