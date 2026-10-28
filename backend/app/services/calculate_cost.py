#TODO
# {
#   "setup_minutes": null,
#   "idle_minutes": null
# }


training_cost = training_hours * gpu_count * price_per_gpu_hour
job_cost = (setup_hours + training_hours + idle_hours) * gpu_count * price_per_gpu_hour