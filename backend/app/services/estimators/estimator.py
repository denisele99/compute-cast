from backend.app.services.estimators.gpu_memory import estimate_memory
from backend.app.services.estimators.gpu_runtime import estimate_runtime
from backend.app.services.estimators.pricing import estimate_cost

#Central estimation function that takes in the data and returns the estimated memory, runtime, and cost

def run_estimation(data):

    memory = estimate_memory(
        parameters=data.parameters
    )

    runtime = estimate_runtime(
        batch_size=data.batch_size,
        epochs=data.epochs
    )

    cost = estimate_cost(
        runtime_hours=runtime,
        hourly_price=data.gpu_hourly_price
    )

    return {
        "memory_gb": memory,
        "runtime_hours": runtime,
        "estimated_cost": cost
    }