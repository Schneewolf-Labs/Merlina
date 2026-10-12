"""Worker-side S3 transport using short-lived, job-scoped signed URLs.

Only the broker receives the worker credential. Signed data requests use a
separate session and never forward that credential to the object store.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import requests


class CapabilityS3Client:
    def __init__(self, broker_url, token, *, session=None, data_session=None):
        parsed = urlsplit(broker_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Artifact broker requires a credential-free HTTPS URL")
        if not token:
            raise ValueError("Artifact broker credential is missing")
        self.url = broker_url
        self.token = token
        self.session = session or requests.Session()
        self.data_session = data_session or requests.Session()

    def _capability(self, key, operation, size=None):
        try:
            response = self.session.post(
                self.url, headers={"Authorization": f"Bearer {self.token}"},
                json={"key": key, "operation": operation, "size": size},
                timeout=30, allow_redirects=False,
            )
            if response.status_code != 200:
                raise ValueError("broker denied capability")
            result = response.json()
            parsed = urlsplit(result["url"])
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("invalid capability URL")
            return result["url"]
        except Exception:
            raise OSError("Could not obtain artifact access") from None

    def _put(self, key, data, size):
        url = self._capability(key, "put", size)
        try:
            response = self.data_session.put(url, data=data, headers={"Content-Length": str(size)},
                                             timeout=(30, 300), allow_redirects=False)
            if not 200 <= response.status_code < 300:
                raise OSError("upload failed")
        except Exception:
            raise OSError("Artifact upload failed") from None

    def put_file(self, local_path, key):
        with open(local_path, "rb") as source:
            self._put(key, source, Path(local_path).stat().st_size)

    def pipe_file(self, key, data):
        self._put(key, data, len(data))

    def _get(self, key, headers=None):
        url = self._capability(key, "get")
        try:
            response = self.data_session.get(url, headers=headers or {}, stream=True,
                                             timeout=(30, 300), allow_redirects=False)
            if response.status_code == 404:
                response.close()
                raise FileNotFoundError("Artifact does not exist")
            if response.status_code not in (200, 206):
                response.close()
                raise OSError("Artifact download failed")
            return response
        except FileNotFoundError:
            raise
        except Exception:
            raise OSError("Artifact download failed") from None

    def get_file(self, key, local_path):
        try:
            with self._get(key) as response, open(local_path, "wb") as destination:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    destination.write(chunk)
        except FileNotFoundError:
            raise
        except Exception:
            raise OSError("Artifact download failed") from None

    def cat_file(self, key, start=None, end=None):
        # This operation is used for manifests, never for model tensors.
        start = 0 if start is None else start
        end = start + 16 * 1024 * 1024 if end is None else end
        if start < 0 or end <= start or end - start > 16 * 1024 * 1024 + 1:
            raise ValueError("Invalid artifact manifest range")
        with self._get(key, {"Range": f"bytes={start}-{end - 1}"}) as response:
            if start and response.status_code != 206:
                raise OSError("Object store ignored range request")
            output = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                output.extend(chunk)
                if len(output) >= end - start:
                    return bytes(output[:end - start])
            return bytes(output)
