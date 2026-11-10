

from abc import ABC, abstractmethod
import os
from typing import override

from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter

SESSION_DIRECTORY = ".sessions"


class SessionManager(ABC):

    @abstractmethod
    def load(self) -> list[ModelMessage]:
        pass


    @abstractmethod
    def save(self, history: list[ModelMessage]) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

class NoOpSessionManager(SessionManager):

    @override
    def load(self) -> list[ModelMessage]:
        return []

    @override
    def save(self, history: list[ModelMessage]) -> None:
        pass

    @override
    def clear(self) -> None:
        pass


class FileSessionManager(SessionManager):

    def __init__(self, session_name: str) -> None:
        self.session_name: str = session_name

    def _session_history_path(self) -> str:
        """Return the session-history path for the given model selection."""

        return os.path.join(
            SESSION_DIRECTORY,
            f"{self.session_name}.json",
        )


    @override
    def load(self) -> list[ModelMessage]:
        """Load session history from disk, falling back to a legacy location."""

        path = self._session_history_path()
        if not os.path.exists(path):
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                history_json = f.read()
            return ModelMessagesTypeAdapter.validate_json(history_json)
        except Exception:
            raise


    @override
    def save(self, history: list[ModelMessage]) -> None:
        """Persist a session history to disk."""
        history_blob = ModelMessagesTypeAdapter.dump_json(history)
        path = self._session_history_path()

        # Create the session directory if it doesn't exist
        os.makedirs(SESSION_DIRECTORY, exist_ok=True)
        with open(path, "wb") as f:
            _ = f.write(history_blob)


    @override
    def clear(self) -> None:
        """Delete stored session history files.

        Returns a tuple of (deleted_any, errors).
        """
        path = self._session_history_path()
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except OSError as _exc:
            raise

