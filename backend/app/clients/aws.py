"""AWS hardware/pricing adapter placeholder. No AWS calls are made."""


def list_gpu_instances(region: str) -> list[dict]:
    # TODO: query provider metadata and return normalized instance details.
    raise NotImplementedError("AWS hardware lookup is not implemented.")


def get_hourly_price(instance_id: str, region: str) -> float:
    # TODO: return a dated, region-specific USD rate with source metadata.
    raise NotImplementedError("AWS pricing lookup is not implemented; enter a rate manually.")
