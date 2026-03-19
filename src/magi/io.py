
from abc import ABC, abstractmethod
import sys
from typing_extensions import override

# Output types

# Output type for the final result of the agent's reasoning process. This is
# the only output type that should be written to standard out. Ideally, this
# should be the only output type that is written to standard out, and all other
# output types should be written to standard error, but this is not strictly
# required. The main requirement is that the final result of the agent's
# reasoning process is written to standard out, and all other output types are
# written to standard error or suppressed entirely.
OTYPE_RESULT = "result"

# Output type for error messages. This should be written to standard error.
OTYPE_ERROR = "error"

# Output type for intermediate thinking steps. This should either be written to
# standard error or standard out, or suppressed entirely, depending on the
# implementation of the outputter.
OTYPE_THINKING = "thinking"

# Output type for promting the user for input. This should be written to
# standard error.
OTYPE_PROMPT = "prompt"

class InputOutput(ABC):
    @abstractmethod
    def read(self) -> str|None:
        """Read input data."""
        pass

    @abstractmethod
    def readapproval(self) -> bool:
        """Read approval from the user."""
        pass

    @abstractmethod
    def write(self, otype: str, data: str) -> None:
        """Write output data to standard out."""
        pass

    @abstractmethod
    def writeln(self, otype: str, data: str) -> None:
        """Write output data to standard out with a newline."""
        pass

class Reader(ABC):
    @abstractmethod
    def read(self) -> str|None:
        """Read input data."""
        pass

class Approver(ABC):
    @abstractmethod
    def readapproval(self) -> bool:
        """Read approval from the user."""
        pass
class Writer(ABC):
    @abstractmethod
    def write(self, otype: str, data: str) -> None:
        """Write output data to standard out."""
        pass

    @abstractmethod
    def writeln(self, otype: str, data: str) -> None:
        """Write output data to standard out with a newline."""
        pass


class ReaderWriter(InputOutput):
    def __init__(self, reader: Reader, writer: Writer, approver: Approver) -> None:
        self._reader: Reader = reader
        self._writer: Writer = writer
        self._approver: Approver = approver

    @override
    def read(self) -> str|None:
        return self._reader.read()

    @override
    def readapproval(self) -> bool:
        return self._approver.readapproval()


    @override
    def write(self, otype: str, data: str) -> None:
        self._writer.write(otype, data)

    @override
    def writeln(self, otype: str, data: str) -> None:
        self._writer.writeln(otype, data)

    @staticmethod
    def console() -> "ReaderWriter":
        return ReaderWriter(ConsoleInputter(), ConsoleOutputter(), ConsoleApprover())

    @staticmethod
    def console_always_approve() -> "ReaderWriter":
        return ReaderWriter(ConsoleInputter(), ConsoleOutputter(), AlwaysApprove())

    @staticmethod
    def non_interactive() -> "ReaderWriter":
        return ReaderWriter(FailInputter(), ConsoleOutputter(), FailApprover())

class FailInputter(Reader):
    @override
    def read(self) -> str|None:
        raise RuntimeError("Input is not supported in this context.")

class FailApprover(Approver):
    @override
    def readapproval(self) -> bool:
        raise RuntimeError("Approval is not supported in this context.")

class ConsoleInputter(Reader):
    @override
    def read(self) -> str|None:
        res = input()
        if res == "":
            return None
        return res

class ConsoleApprover(Approver):
    @override
    def readapproval(self) -> bool:
        while True:
            response = input().strip().lower()
            if response in ("y", "yes"):
                return True
            elif response in ("n", "no"):
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")


class AlwaysApprove(Approver):
    @override
    def readapproval(self) -> bool:
        return True

class ConsoleOutputter(Writer):
    @override
    def write(self, otype: str, data: str) -> None:
        if otype != OTYPE_RESULT:
            _ = sys.stderr.write(f"{data}")
            _ = sys.stderr.flush()
        else:
            _ = sys.stdout.write(f"{data}")
            _ = sys.stdout.flush()

    @override
    def writeln(self, otype: str, data: str) -> None:
        self.write(otype, f"{data}\n")

