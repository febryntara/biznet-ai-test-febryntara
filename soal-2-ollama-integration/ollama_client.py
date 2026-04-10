"""
Client untuk berkomunikasi dengan Ollama API.
Menggunakan model LLM lokal yang sudah terinstall di Ollama.
"""

import requests
import json
from typing import Optional, Dict, Any


class OllamaClient:
    """Client untuk berinteraksi dengan Ollama API."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma3:1b"):
        """
        Initialize Ollama client.
        
        Args:
            base_url: URL Ollama API (default: http://localhost:11434)
            model: Model name to use (default: gemma3:1b)
        """
        self.base_url = base_url
        self.model = model
        self.api_url = f"{base_url}/api"
        
    def check_connection(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except Exception:
            return False
    
    def get_available_models(self) -> list:
        """Get list of available models in Ollama."""
        try:
            response = requests.get(f"{self.api_url}/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except Exception as e:
            print(f"Error getting models: {e}")
            return []
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate response from Ollama model.
        
        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt for context
            
        Returns:
            Dictionary with response data or error information
        """
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.api_url}/generate",
                json=payload,
                timeout=120  # 120 second timeout for generation
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "response": data.get("response", ""),
                    "model": data.get("model", self.model),
                    "total_duration": data.get("total_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "prompt_eval_count": data.get("prompt_eval_count", 0),
                    "eval_count": data.get("eval_count", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
                
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Cannot connect to Ollama. Make sure Ollama is running."
            }
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timeout. Model might be taking too long to respond."
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def chat(self, messages: list, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Chat completion with context/message history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with response data
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "success": True,
                    "message": data.get("message", {}),
                    "model": data.get("model", self.model),
                    "total_duration": data.get("total_duration", 0)
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error: {str(e)}"
            }


# Default client instance
default_client = OllamaClient()


def get_response(prompt: str, system_prompt: Optional[str] = None) -> str:
    """
    Simple helper function to get response from Ollama.
    
    Args:
        prompt: User prompt
        system_prompt: Optional system prompt
        
    Returns:
        Response text or error message
    """
    client = OllamaClient()
    result = client.generate_response(prompt, system_prompt)
    
    if result["success"]:
        return result["response"]
    else:
        return f"Error: {result.get('error', 'Unknown error')}"


if __name__ == "__main__":
    # Test the client
    client = OllamaClient()
    
    print("Testing Ollama connection...")
    if client.check_connection():
        print("✓ Ollama is running")
        
        models = client.get_available_models()
        print(f"Available models: {[m['name'] for m in models]}")
        
        # Test simple generation
        print("\nTesting generation...")
        test_prompt = "Hello, how are you?"
        result = client.generate_response(test_prompt)
        
        if result["success"]:
            print(f"Model: {result['model']}")
            print(f"Response: {result['response']}")
            print(f"Duration: {result['total_duration'] / 1e9:.2f}s")
        else:
            print(f"Error: {result.get('error')}")
    else:
        print("✗ Cannot connect to Ollama. Make sure Ollama is running.")