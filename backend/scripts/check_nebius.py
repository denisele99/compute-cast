"""Read-only check of the benchmark Nebius settings. Creates nothing and costs nothing.

    cd backend && python scripts/check_nebius.py

Uses the same settings as the backend (environment variables or backend/.env) and:
  1. signs in as the job-submitter service account and shows which account it is;
  2. lists Serverless AI jobs in NEBIUS_PROJECT_ID (proves the account can see jobs there);
  3. tries to download a non-existent object from the results bucket with the results-reader key.
Secrets are never printed. Needs: pip install -r requirements-cloud.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config  # noqa: E402,F401  (loads backend/.env)
from app.services.benchmarks.policy import load_policy  # noqa: E402

LIMIT = {'timeout': 30, 'auth_timeout': 30, 'retries': 1}  # fail fast instead of hanging


def step(name, fn):
    try:
        detail = fn()
        print(f'  OK    {name}' + (f': {detail}' if detail else ''))
        return True
    except Exception as exc:  # report the failure, never the credentials
        print(f'  FAIL  {name}: {type(exc).__name__}: {str(exc).splitlines()[0][:300]}')
        return False


def main():
    p = load_policy()
    print('Nebius benchmark settings check (read-only)')
    missing = [m for m in p.missing_for_cloud() if m.startswith(('NEBIUS_', 'BENCH_IMAGE'))] if p.backend == 'sdk' else []
    if p.credentials_file:
        print(f'  using NEBIUS_CREDENTIALS_FILE ({p.credentials_file})')
        if not Path(p.credentials_file).is_file():
            print('  FAIL  credentials file not found at that path (use the full path, e.g. /home/you/.nebius/credentials.json)')
            return 1
    elif p.service_account_id:
        print(f'  using authorized key for {p.service_account_id}')
        if p.private_key_file and not Path(p.private_key_file).is_file():
            print(f'  FAIL  private key not found at {p.private_key_file} (use the full path to private.pem)')
            return 1
    if missing:
        print('  missing settings: ' + ', '.join(missing))
    if not p.project_id or not (p.credentials_file or (p.service_account_id and p.public_key_id and p.private_key_file)):
        print('  FAIL  set NEBIUS_PROJECT_ID and NEBIUS_CREDENTIALS_FILE (or the three authorized-key settings)')
        return 1

    from nebius.api.nebius.ai.v1 import JobServiceClient, ListJobsRequest
    from nebius.api.nebius.iam.v1 import GetProfileRequest, ProfileServiceClient
    from app.services.benchmarks.nebius_client import NebiusSDKClient
    client = NebiusSDKClient(p)
    # Build the SDK with the backend's exact sign-in settings (reads the key file).
    if not step('load credentials', lambda: client._service() and None):
        print('Fix the credentials file (or key settings) and run again.')
        return 1
    sdk = client._sdk

    def whoami():
        profile = ProfileServiceClient(sdk).get(GetProfileRequest(), **LIMIT).wait()
        sa = getattr(profile, 'service_account_profile', None)
        info = getattr(sa, 'info', None) if sa is not None else None
        meta = getattr(info, 'metadata', None)
        return f"service account {getattr(meta, 'name', '?')} ({getattr(meta, 'id', '?')})" if meta else 'signed in'

    def jobs():
        page = JobServiceClient(sdk).list(ListJobsRequest(parent_id=p.project_id, page_size=5), **LIMIT).wait()
        return f'{len(page.items)} job(s) visible in {p.project_id}'

    ok = step('sign in', whoami)
    ok = step('list jobs in the project', jobs) and ok

    if p.results_bucket_name and p.s3_access_key_id and p.s3_secret_access_key:
        def bucket():
            import boto3
            s3 = boto3.client('s3', endpoint_url=p.s3_endpoint, region_name=p.s3_region,
                              aws_access_key_id=p.s3_access_key_id, aws_secret_access_key=p.s3_secret_access_key)
            from botocore.exceptions import ClientError
            try:  # download test on a key that does not exist (the reader may download, not list)
                s3.get_object(Bucket=p.results_bucket_name, Key=f'{p.results_prefix}/_compute_cast_check/none.json')
            except ClientError as exc:
                code = exc.response.get('Error', {}).get('Code')
                if code in ('NoSuchKey', '404'):
                    return f'bucket {p.results_bucket_name} reachable with the reader key'
                raise RuntimeError(f'{code}: check the bucket name, endpoint/region and that the reader has '
                                   'storage.object-viewer on the bucket') from None
            return f'bucket {p.results_bucket_name} reachable'
        ok = step('read results bucket', bucket) and ok
    else:
        print('  SKIP  results bucket (set NEBIUS_RESULTS_BUCKET_NAME and the reader access key)')
    print('All checks passed.' if ok else 'Some checks failed; see above.')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
