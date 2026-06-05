from enum import Enum
from typing import List
import logging

from pydantic import BaseModel

from ai_client import create_ai_client, set_pricing_file
from decouple import config

# Enable debug logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

API_KEY="sk-ws-djI.BdR1fEB5Y6hunu4jb2egTidcv3d26wPs2FAcvp8A3PGYSIUsViaX1PeLnTYdgaAUj8VZ4xzzDnAW1bXScSIDhzSP9wTDwMEs0QYgahn3q02PN-X11VV07hDbsTLNZYms.MEQCIDtQOQqGWRq4Eo2CqRRGCkoAvP3ocaFaDBXoIAqLbpGPAiBKXZEvUwigAQcpZhh5pRFUguzueEuXLMgIwixyME9jSQ"
# Create a client for any provider
client = create_ai_client('alibaba', api_key=API_KEY,
                          base_url="https://ws-flpcowcdt4x02z3b.eu-central-1.maas.aliyuncs.com/compatible-mode/v1",
                          max_retries=0,
                          )


response1 = client.prompt(
    'qwen3.5-plus',
    "What's on the image?",
    images=['wordle.png']
)

print(response1.text)
print(response1.parsed)
