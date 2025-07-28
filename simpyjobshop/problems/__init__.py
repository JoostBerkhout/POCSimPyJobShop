# Import the base/core dependencies first
from .MachineSchedule import MachineSchedule  # isort: skip
from .Problem import Problem  # isort: skip

# Import the problem-specific classes
from .FlexibleJobShop import (
    FlexibleJobShop,
    FlexibleJobShopBase,
    FlexibleJobShopFullStoch,
    FlexibleJobShopFullStochNoTard,
    FlexibleJobShopNoTard,
)
from .HybridFlowShop import (
    HybridFlowShop,
    HybridFlowShopBase,
    HybridFlowShopFullStoch,
    HybridFlowShopFullStochNoTard,
    HybridFlowShopNoTard,
)
from .JobShop import (
    JobShop,
    JobShopBase,
    JobShopFullStoch,
    JobShopFullStochNoTard,
    JobShopNoTard,
)
from .JobShopEasy import JobShopEasy
from .OpenShop import (
    OpenShop,
    OpenShopBase,
    OpenShopFullStoch,
    OpenShopFullStochNoTard,
    OpenShopNoTard,
)
from .OpenShopEasy import OpenShopEasy
from .ParallelMachines import (
    ParallelMachines,
    ParallelMachinesBase,
    ParallelMachinesFullStoch,
    ParallelMachinesFullStochNoTard,
    ParallelMachinesNoTard,
)
