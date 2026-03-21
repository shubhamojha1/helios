from dataclasses import dataclass 
from typing import Optional

@dataclass
class Page:
    page_id: int
    is_free: bool = True
    owner_request_id: Optional[str] = None
    token_offset: int = 0
    ref_count: int = 0
