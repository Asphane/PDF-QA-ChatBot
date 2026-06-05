"""
chat_history.py

Provides the ChatHistoryMsg class and Message dataclass to manage chat session state
and convert application messages to LangChain-compatible message formats.
"""

from dataclasses import dataclass, field
from typing import Literal

@dataclass
class Message:
    """
    Represents a single chat message in the session.
    
    Attributes:
        role: The role of the sender ('user' or 'assistant').
        content: The text content of the message.
        sources: A list of source citations for the message.
    """
    role: Literal['user', 'assistant']
    content: str
    sources: list = field(default_factory=list)

class ChatHistoryMsg:
    """
    Manages the session chat history, storing messages and converting them
    for LangChain use.
    """
    def __init__(self):
        """Initializes the message list."""
        self.messages: list[Message] = []

    def add(self, role, content, sources=None):
        """
        Adds a new message to the chat history.
        
        Args:
            role: The role of the sender.
            content: The text content of the message.
            sources: Optional list of sources.
        """
        self.messages.append(Message(role, content, sources or []))
        
    def langchain_msgs(self):
        """
        Convert to LangChain HumanMessage/AIMessage for memory-aware QA.
        
        Returns:
            List of LangChain HumanMessage or AIMessage objects.
        """
        from langchain_core.messages import HumanMessage, AIMessage

        res = []
        for m in self.messages:
            if m.role == 'user':
                res.append(HumanMessage(content = m.content))

            if m.role == 'assistant':
                res.append(AIMessage(content = m.content))

        return res