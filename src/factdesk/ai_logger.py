"""Provide the application logger and its one-time setup."""

#==========================================================================
# Imports
#==========================================================================
from    __future__                      import  annotations

import  logging
import  sys


aiLogger = logging.getLogger("factdesk")


#==========================================================================
# Configure the logger once
#==========================================================================
def set_up_ai_logger(level: str = "INFO") -> None:
    """
    Send the application logger to stderr with a validated level; stdout
    stays free for command output.

    Input:
        . level         : DEBUG, INFO, WARNING, ERROR or CRITICAL
    """
    if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(f"unknown log level: {level}")
    aiLogger.setLevel(level)
    if not aiLogger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        aiLogger.addHandler(handler)
    aiLogger.propagate = False
