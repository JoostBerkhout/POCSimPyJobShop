import bisect


class MachineSchedule:
    """
    Represents the schedule (timeline) of a single machine.

    Attributes
    ----------
    timeline : list of tuple
        List of (start, end) times of tasks scheduled on this machine,
        sorted by start time.
    """

    def __init__(self):
        """Initialize an empty machine schedule."""
        self.timeline = []  # List of (start, end) tuples

    def find_earliest_slot(self, earliest_ready, duration):
        """
        Find the earliest available time slot on this machine for a task.

        Parameters
        ----------
        earliest_ready : int or float
            The earliest time the task is allowed to start.
        duration : int or float
            The duration of the task to schedule.

        Returns
        -------
        int or float
            The earliest start time of the task on this machine.
        """
        if not self.timeline:
            return earliest_ready

        if earliest_ready + duration <= self.timeline[0][0]:
            return earliest_ready

        for i in range(len(self.timeline) - 1):
            start_in_gap = max(self.timeline[i][1], earliest_ready)
            gap_end = self.timeline[i + 1][0]
            if gap_end - start_in_gap >= duration:
                return start_in_gap

        return max(self.timeline[-1][1], earliest_ready)

    def add_task(self, start, end):
        """
        Add a task to the machine schedule.

        Parameters
        ----------
        start : int or float
            Start time of the task.
        end : int or float
            End time of the task (exclusive).
        """
        bisect.insort(self.timeline, (start, end))
