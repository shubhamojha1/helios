from memory.manager import MemoryManager, AllocationError
import pytest

# From project root:
# python -m pytest tests/memory/test_memory.py -v

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

def test_free():
    mm = MemoryManager(total_pages=20, page_size_tokens=16)
    mm.allocate("req_1", 40)  # 3 pages
    mm.allocate("req_2", 32)  # 2 pages
    mm.free("req_1")

    used, total = mm.get_utilization()
    assert used == 2


def test_can_allocate():
    mm = MemoryManager(total_pages=20, page_size_tokens=16)
    assert mm.can_allocate(100) == True
    assert mm.can_allocate(99999) == False

def test_allocation_error():
    mm = MemoryManager(total_pages=5, page_size_tokens=16)
    with pytest.raises(AllocationError):
        mm.allocate("req_1", 99999)

def test_reallocate_after_free():
    mm = MemoryManager(total_pages=20, page_size_tokens=16)
    mm.allocate("req_1", 100)  # 7 pages
    mm.free("req_1")
    mm.allocate("req_2", 100)  # should work fine

    used, total = mm.get_utilization()
    assert used == 7

# def test_get_fragmentation_ratio():
#     mm = MemoryManager(total_pages=20, page_size_tokens=16)
#     mm.allocate("req_1", 100)
#     mm.allocate("req_2", 17)

#     fragmentation_ratio = mm.get_fragmentation_ratio()