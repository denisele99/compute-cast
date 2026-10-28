from backend.app.clients.nebius import launch_gpu_job


def submit_benchmark(request):
    # Validate workload and GPU selection

    # Prepare benchmark configuration

    # Submit job to Nebius
    job = launch_gpu_job(request)

    return {
        "job_id": job["id"],
        "status": "submitted"
    }