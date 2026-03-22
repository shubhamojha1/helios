from core.types import Page
from core.logger import get_logger
from typing import List, Dict, Tuple
import math

logger = get_logger(__name__)

"""
What it needs to do:

Initialize a fixed pool of Page objects (all free at start)
can_allocate(num_tokens) — checks if enough free pages exist
allocate(request_id, num_tokens) — marks pages as owned by a request, returns their IDs
extend(request_id) — called each decoding step; only allocates a new page when the current last page is full (lazy allocation)
free(request_id) — releases all pages owned by a request back to the pool
get_utilization() — returns (pages_used, pages_total)
get_fragmentation_ratio() — ratio of wasted slots within allocated pages

How to think about it: 
It's just a pool manager. Keep a list of all pages, a set of free page IDs, and a dict mapping request_id to list of page IDs. No GPU interaction, pure Python bookkeeping.
Test it standalone before moving on — write a small script that allocates pages for 5 fake requests, frees two of them, allocates again, and prints utilization. Make sure numbers add up.
"""

class AllocationError(Exception):
    pass

class MemoryManager:
    def __init__(self, total_pages: int, page_size_tokens: int):
        self.total_pages = total_pages
        self.page_size_tokens = page_size_tokens

        # initialize all pages as free
        self.pages: Dict[int, Page] = {
            i: Page(page_id=i) for i in range(total_pages)
        }

        self.free_page_ids: set = set(range(total_pages))

        # Maps request id -> list of page_ids allocated to it
        self.request_pages: Dict[str, List[int]] = {}

    def _pages_needed(self, num_tokens: int) -> int:
        return math.ceil(num_tokens / self.page_size_tokens)
    
    def can_allocate(self, num_tokens: int) -> bool:
        return len(self.free_page_ids) >= self._pages_needed(num_tokens)
    
    def allocate(self, request_id: str, num_tokens: int) -> List[int]:
        needed = self._pages_needed(num_tokens)

        if len(self.free_page_ids) < needed:
            logger.warning(f"Allocation failed for request {request_id}: needed {needed}, free {len(self.free_page_ids)}")
            raise AllocationError(
                f"Cannot allocate {needed} pages for request {request_id}, "
                f"only {len(self.free_page_ids)} free"
            )
        
        # Pick 'needed' pages from the free pool
        allocated = []
        for _ in range(needed):
            page_id = next(iter(self.free_page_ids))
            self.free_page_ids.remove(page_id)

            page = self.pages[page_id]
            page.is_free = False
            page.owner_request_id = request_id
            page.token_offset = len(allocated) * self.page_size_tokens

            allocated.append(page_id)

        self.request_pages[request_id] = allocated
        logger.debug(f"Allocated {needed} pages for request {request_id}")
        return allocated
    
    def get_utilization(self) -> Tuple[int, int]:
        used = self.total_pages - len(self.free_page_ids)
        return used, self.total_pages

    def free(self, request_id: str) -> None:
        if request_id not in self.request_pages:
            return
        
        # One of more pages could be tied to a single request
        # Need to free all of them
        for page_id in self.request_pages[request_id]:
            page = self.pages[page_id]
            page.is_free = True
            page.owner_request_id = None
            page.token_offset = 0
            self.free_page_ids.add(page_id)

        del self.request_pages[request_id]
        logger.debug(f"Freed all pages for request {request_id}")

    def get_fragmentation_ratio(self) -> float:
        """
        Fragmentation = wated slots in the last page of each request.
        A request using 17 tokens with page_size=16
        Uses 1 page for first 16 tokens, and 1 page for last token
        Wastage of 15 slots in the 2nd page
        """
        if not self.request_pages:
            return 0.0
        
        total_allocated_slots = 0
        total_wasted_slots = 0

        for request_id, page_ids in self.request_pages.items():
            allocated_slots = len(page_ids) * self.page_size_tokens
            total_allocated_slots += allocated_slots
            # We don't track exact token count per request
            # So fragmentation is approximated as 0 for now.
            # Will be updated when scheduler tracks token counts.

        if total_allocated_slots == 0:
            return 0.0
        
        return total_wasted_slots / total_allocated_slots


if __name__ == "__main__":
    manager = MemoryManager()
    print(manager.Pages)