import os
import logging
import google.generativeai as genai
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("GeminiAgentService")

class GeminiAgent:
    """
    Interfaces with Google Gemini API to provide contextual reasoning for RAG.
    Supports Gemini 2.5 Pro / 1.5 Pro / 1.5 Flash models as specified in system plan.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = 'gemini-1.5-flash'):
        # Fallback to env var if key not explicitly passed
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key or key.startswith("YOUR_"):
            logger.warning("No valid Gemini API key provided. Gemini Agent will operate in mock mode.")
            self.model = None
            return

        try:
            genai.configure(api_key=key)
            self.model_name = model_name
            self.model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=(
                    "You are BlindAssist, a supportive, clear, and concise AI assistant for visually impaired students. "
                    "You will be provided with NCERT textbook context extracted via RAG. "
                    "Use the provided context to answer questions accurately in 2-3 simple sentences suitable for speech synthesis. "
                    "If the context does not contain the answer, state honestly that the information wasn't found in the text."
                )
            )
            logger.info(f"Gemini Agent successfully initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini Agent: {e}")
            self.model = None

    def generate_response(self, query: str, context: str) -> str:
        """
        Generates a contextual response based on user query and retrieved RAG context.
        """
        if not query:
            return "No question received."

        if not self.model:
            return f"[Offline Mock Response] Based on context: '{context[:100]}...', here is the answer for '{query}'."

        # Construct augmented prompt
        prompt = f"NCERT Document Context:\n{context}\n\nUser Question: {query}"

        try:
            response = self.model.generate_content(prompt)
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            return "I'm sorry, I couldn't generate a clear answer from the document."
        except Exception as e:
            logger.error(f"Gemini API Error during generation: {e}")
            return f"Gemini service encountered an error: {e}"

