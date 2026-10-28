#TODO
# {
#   "setup_minutes": null,
#   "idle_minutes": null
# }

def estimate_cost(training_hours, gpu_count, price_per_gpu_hour, setup_hours=0, idle_hours=0):
    """
    Estimate the cost of a job based on training hours, GPU count, and price per GPU hour.
    
    Parameters:
    - training_hours (float): The number of hours spent on training.
    - gpu_count (int): The number of GPUs used.
    - price_per_gpu_hour (float): The cost per GPU per hour.
    - setup_hours (float): The number of hours spent on setup (default is 0).
    - idle_hours (float): The number of hours spent idle (default is 0).
    
    Returns:
    - float: The estimated total cost of the job.
    """
    
    #training_cost = training_hours * gpu_count * price_per_gpu_hour
    job_cost = (setup_hours + training_hours + idle_hours) * gpu_count * price_per_gpu_hour

    return job_cost
