import os
import logging
import google.generativeai as genai
from typing import Optional

logger = logging.getLogger("GeminiAgentService")

class GeminiAgent:
    """
    Interfaces with Google Gemini API to provide contextual reasoning.
    """
    def __init__(self, api_key: str, model_name: str = 'gemini-1.5-flash'):
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=(
                    "You are BlindAssist, a supportive and concise AI assistant for visually impaired users. "
                    "You will be provided with text extracted from a document via OCR. "
                    "Use the provided context to answer the user's questions accurately. "
                    "If the context does not contain the answer, be honest and say you don't know "
                    "or suggest what they might look for. Keep responses natural and easy to listen to via TTS."
                )
            )
            logger.info(f"Gemini Agent initialized with model {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Agent: {e}")
            raise

    def generate_response(self, query: str, context: str) -> Optional[str]:
        """
        Generates a response based on the user query and retrieved RAG context.
        """
        if not query:
            return None

        # Construct the augmented prompt
        prompt = f"Context from scanned document:\n{context}\n\nUser Question: {query}"

        try:
            # Using a simple generation call for the MVP
            response = self.model.generate_content(prompt)

            if response.text:
                return response.text.strip()

            return "I'm sorry, I couldn't generate a response."
        except Exception as e:
            logger.error(f"Gemini API Error: {e}")
            return f"I encountered an error while processing your request: {e}"
