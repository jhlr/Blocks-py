from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Union




###############################################################################
# Exceptions for internal control flow
###############################################################################

class BlockReturn(BaseException):
	"""Signal a Block-level return (top-level exit)."""
	pass




class LoopBreak(BaseException):
	"""Signal a break in the current loop."""
	pass




class LoopContinue(BaseException):
	"""Signal a continue in the current loop."""
	pass
