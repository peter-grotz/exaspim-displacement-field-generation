"""Stage inputs from S3 to local disk for the finalize pipeline.

Two inputs come from S3:
  * a TRANSFORM FOLDER (an s3:// prefix) holding the manual Slicer Transform*.h5 files, and
  * an INPUT VOLUME (a single s3:// object, e.g. <sample>_10um_to_ccf_moved.nii.gz).

aind-open-data is a public bucket, so we read ANONYMOUSLY by default (no credentials needed,
matching the other exaSPIM capsules). If AWS credentials are present and anonymous access is
denied, we transparently retry signed. Downloads are skipped if the local copy already exists
with the same size (idempotent re-runs).
"""
from __future__ import annotations

import os
from typing import List, Tuple

# NOTE: boto3/botocore are imported lazily (inside the functions that hit S3), so importing this
# module — and running the finalize pipeline against LOCAL/attached /data inputs — does NOT
# require boto3 to be installed. Only actual s3:// downloads pull it in.


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    """'s3://bucket/some/key' -> ('bucket', 'some/key')."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    rest = uri[len("s3://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def is_s3(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")


def _client(anonymous: bool):
    import boto3
    from botocore import UNSIGNED
    from botocore.client import Config
    if anonymous:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def _with_fallback(fn):
    """Try anonymous first; on AccessDenied/403 retry with signed credentials."""
    from botocore.exceptions import ClientError
    try:
        return fn(_client(anonymous=True))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("AccessDenied", "403", "InvalidAccessKeyId", "AuthorizationHeaderMalformed"):
            return fn(_client(anonymous=False))
        raise


def _download_object(bucket: str, key: str, dest: str) -> str:
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)

    def head(c):
        return c.head_object(Bucket=bucket, Key=key)["ContentLength"]

    size = _with_fallback(head)
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        print(f"  [s3] cached {dest} ({size:,} B)")
        return dest
    print(f"  [s3] get s3://{bucket}/{key} -> {dest} ({size:,} B)")
    _with_fallback(lambda c: c.download_file(bucket, key, dest))
    return dest


def _list_keys(bucket: str, prefix: str) -> List[str]:
    def run(c):
        keys, token = [], None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = c.list_objects_v2(**kw)
            for o in resp.get("Contents", []):
                keys.append(o["Key"])
            if not resp.get("IsTruncated"):
                return keys
            token = resp.get("NextContinuationToken")

    return _with_fallback(run)


def stage_volume(uri: str, dest_dir: str) -> str:
    """Download a single S3 object into dest_dir; return the local path."""
    bucket, key = parse_s3_uri(uri)
    return _download_object(bucket, key, os.path.join(dest_dir, os.path.basename(key)))


def stage_transforms(uri: str, dest_dir: str, suffix: str = ".h5") -> str:
    """Download every ``suffix`` object under the S3 prefix into dest_dir; return that dir.

    Filenames are flattened to their basenames (the transform folder is flat in practice).
    """
    bucket, prefix = parse_s3_uri(uri)
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    keys = [k for k in _list_keys(bucket, prefix)
            if k.lower().endswith(suffix) and not k.endswith("/")]
    if not keys:
        raise FileNotFoundError(f"no {suffix} objects under s3://{bucket}/{prefix}")
    os.makedirs(dest_dir, exist_ok=True)
    for k in keys:
        _download_object(bucket, k, os.path.join(dest_dir, os.path.basename(k)))
    print(f"  [s3] staged {len(keys)} transform(s) -> {dest_dir}")
    return dest_dir
