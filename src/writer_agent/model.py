"""Shared language-model configuration."""

from os import getenv

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

llm = ChatGroq(model=getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL))
