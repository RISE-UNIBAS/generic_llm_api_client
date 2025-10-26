"""
Google Gemini-specific implementation of the BaseAIClient.

This module provides the GeminiClient class, which implements the BaseAIClient
interface specifically for Google's Gemini API, supporting both text-only and
multimodal (text + images) content.
"""
import tempfile
import time
from datetime import datetime
from typing import List, Tuple, Any, Optional

import google.generativeai as genai
import requests

from .base_client import BaseAIClient


class GeminiClient(BaseAIClient):
    """
    Google Gemini-specific implementation of the BaseAIClient.
    
    This class implements the BaseAIClient interface for Google's Gemini API,
    handling both text-only and multimodal (text + images) requests.
    
    Key features:
    - Full support for multimodal content via Gemini Pro Vision models
    - Image uploading specific to Gemini's requirements
    - Support for Gemini-specific parameters like top_k, top_p, etc.
    """
    
    PROVIDER_ID = "genai"
    SUPPORTS_MULTIMODAL = True
    
    def _init_client(self):
        """Configure Gemini API with the provided API key."""
        genai.configure(api_key=self.api_key)
        # No specific client object is needed for Gemini
        self.api_client = True
    
    def _prepare_images(self) -> List[Any]:
        """
        Process and upload image resources for Gemini.
        
        Returns:
            list: List of uploaded image objects
        """
        images = []
        
        for resource in self.image_resources:
            if self.is_url(resource):
                # For URLs, fetch the image and provide as blob
                try:
                    response = requests.get(resource)
                    if response.status_code == 200:
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
                            tmp.write(response.content)
                            tmp.flush()  # Ensure everything is written before upload
                            
                            # Apply resize if specified
                            resize_option = None
                            if 'image_resize' in self.settings and self.settings['image_resize'] not in ('', 'none'):
                                try:
                                    width, height = self.settings['image_resize'].split('x')
                                    resize_option = (int(width), int(height))
                                except (ValueError, TypeError):
                                    # If parsing fails, don't apply resize
                                    pass
                            
                            # Upload the image with optional resize
                            if resize_option:
                                image_part = genai.upload_file(path=tmp.name, resize=resize_option)
                            else:
                                image_part = genai.upload_file(path=tmp.name)
                                
                            images.append(image_part)
                except Exception as e:
                    print(f"Error fetching image from URL {resource}: {e}")
            else:
                # For local files, use the file path
                try:
                    # Apply resize if specified
                    resize_option = None
                    if 'image_resize' in self.settings and self.settings['image_resize'] not in ('', 'none'):
                        try:
                            width, height = self.settings['image_resize'].split('x')
                            resize_option = (int(width), int(height))
                        except (ValueError, TypeError):
                            # If parsing fails, don't apply resize
                            pass
                    
                    # Upload the image with optional resize
                    if resize_option:
                        image_file = genai.upload_file(path=resource, resize=resize_option)
                    else:
                        image_file = genai.upload_file(path=resource)
                        
                    images.append(image_file)
                except Exception as e:
                    print(f"Error uploading image file {resource}: {e}")
        
        return images
    
    def do_prompt(self, model: str, prompt: str) -> Any:
        """
        Send a prompt to the Gemini model and get the response.
        
        Args:
            model (str): The Gemini model identifier (e.g., "gemini-pro", "gemini-pro-vision")
            prompt (str): The text prompt to send
            
        Returns:
            tuple: (response_object, elapsed_time_in_seconds)
        """
        # Initialize the model with generation config
        generation_config = {}
        
        # Extract Gemini-specific parameters from settings
        optional_params = [
            'temperature', 'top_p', 'top_k', 'max_output_tokens',
            'candidate_count', 'stop_sequences'
        ]
        
        for param in optional_params:
            if param in self.settings and self.settings[param] is not None:
                # Map max_tokens to max_output_tokens if needed
                if param == 'max_tokens':
                    generation_config['max_output_tokens'] = self.settings[param]
                else:
                    generation_config[param] = self.settings[param]
        
        # Initialize the model
        if not generation_config:
            # Use default config if no custom settings
            gemini_model = genai.GenerativeModel(model)
        else:
            gemini_model = genai.GenerativeModel(model, generation_config=generation_config)
        
        # Process images if available
        images = []
        if self.image_resources:
            images = self._prepare_images()
        
        # Generate content with text and any images
        response = gemini_model.generate_content([prompt] + images)

        return response

    def transpose_response(self, response: Any) -> dict:
        return_value = self.get_empty_generic_response()
        return_value['text'] = response.candidates[0].content.parts[0].text
        return return_value

    def get_model_list(self) -> List[Tuple[str, Optional[str]]]:
        """
        Get a list of available models from Gemini.
        
        Returns:
            list: List of tuples (model_id, created_date)
        """
        model_list = []
        
        raw_list = genai.list_models()
        for model in raw_list:
            if model.display_name.startswith("Gemini"):
                model_list.append((model.name.lstrip("models/"), None))
        
        return model_list