from simpyjobshop.problems import MachineSchedule


def test_empty_schedule():
    m = MachineSchedule()
    assert m.find_earliest_slot(0, 5) == 0


def test_add_and_find_gap():
    m = MachineSchedule()
    m.add_task(2, 5)
    m.add_task(7, 10)
    assert m.find_earliest_slot(0, 2) == 0


def test_forced_end_append():
    m = MachineSchedule()
    m.add_task(0, 4)
    m.add_task(5, 9)
    assert m.find_earliest_slot(0, 3) == 9


def test_start_after_ready_time():
    m = MachineSchedule()
    m.add_task(3, 6)
    m.add_task(8, 10)
    assert m.find_earliest_slot(7, 1) == 7
