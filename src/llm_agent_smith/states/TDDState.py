from datetime import datetime
import json
from typing import Dict, List, TypedDict, Optional, Annotated


class TDDState(TypedDict):
    user_request: str
    features: Annotated[List[str], lambda l1, l2: l1 + l2]
    current_feature: Optional[str]
    production_code: str
    test_code: str
    test_results: Optional[str]
    history: Annotated[List[Dict], lambda l1, l2: l1 + l2]
    iteration_count: int
