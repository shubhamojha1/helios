from memory.manager import MemoryManager, AllocationError
import pytest

def test_basic_allocation():
    mm = MemoryManager(total_pages=20, page_size_tokens=16)
    mm.allocate("req_1", 10)
    mm.allocate("req_2", 40)
    mm.allocate("req_3", 16)
    mm.allocate("req_4", 100)
    mm.allocate("req_5", 32)

    used, total = mm.get_utilization()
    assert used == 14
    assert total == 20
