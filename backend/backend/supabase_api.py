from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


class SupabaseError(RuntimeError):
    pass


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str

    @classmethod
    def from_service_role_env(cls) -> "SupabaseConfig":
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
        return cls(url=url.rstrip("/"), key=key)

    @classmethod
    def from_anon_env(cls) -> "SupabaseConfig":
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        if not url or not key:
            raise RuntimeError("SUPABASE_URL and SUPABASE_ANON_KEY are required")
        return cls(url=url.rstrip("/"), key=key)


class SupabaseApi:
    def __init__(self, config: SupabaseConfig, *, bearer_token: str | None = None):
        self.config = config
        self.bearer_token = bearer_token or config.key
        self.client = httpx.Client(timeout=120)

    def close(self) -> None:
        self.client.close()

    def rest_headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.config.key,
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }
        if extra:
            headers.update(extra)
        return headers

    def storage_headers(self, content_type: str, *, upsert: bool = True) -> dict[str, str]:
        return {
            "apikey": self.config.key,
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": content_type,
            "x-upsert": "true" if upsert else "false",
        }

    def select_one(self, table: str, query: str) -> dict[str, Any]:
        url = f"{self.config.url}/rest/v1/{table}?{query}"
        res = self.client.get(
            url,
            headers=self.rest_headers({"Accept": "application/vnd.pgrst.object+json"}),
        )
        return self._json(res)

    def patch_rows(
        self,
        table: str,
        query: str,
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        url = f"{self.config.url}/rest/v1/{table}?{query}"
        res = self.client.patch(
            url,
            headers=self.rest_headers({"Prefer": "return=representation"}),
            content=json.dumps(payload),
        )
        return self._json(res)

    def delete_rows(self, table: str, query: str) -> list[dict[str, Any]]:
        url = f"{self.config.url}/rest/v1/{table}?{query}"
        res = self.client.delete(
            url,
            headers=self.rest_headers({"Prefer": "return=representation"}),
        )
        return self._json(res)

    def upsert_rows(
        self,
        table: str,
        payload: list[dict[str, Any]],
        *,
        on_conflict: str,
    ) -> list[dict[str, Any]]:
        url = f"{self.config.url}/rest/v1/{table}?on_conflict={on_conflict}"
        res = self.client.post(
            url,
            headers=self.rest_headers(
                {"Prefer": "resolution=merge-duplicates,return=representation"}
            ),
            content=json.dumps(payload),
        )
        return self._json(res)

    def upload_bytes(
        self,
        bucket: str,
        path: str,
        data: bytes,
        content_type: str,
    ) -> None:
        url = f"{self.config.url}/storage/v1/object/{bucket}/{path.lstrip('/')}"
        res = self.client.post(url, headers=self.storage_headers(content_type), content=data)
        self._raise_for_status(res)

    def upload_json(self, bucket: str, path: str, payload: Any) -> None:
        self.upload_bytes(
            bucket,
            path,
            json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            "application/json",
        )

    def download_bytes(self, bucket: str, path: str) -> bytes:
        url = f"{self.config.url}/storage/v1/object/{bucket}/{path.lstrip('/')}"
        res = self.client.get(url, headers=self.rest_headers())
        self._raise_for_status(res)
        return res.content

    def upload_file(
        self,
        bucket: str,
        path: str,
        file_path: str | Path,
        content_type: str,
    ) -> None:
        self.upload_bytes(bucket, path, Path(file_path).read_bytes(), content_type)

    def _json(self, res: httpx.Response) -> Any:
        self._raise_for_status(res)
        if not res.content:
            return None
        return res.json()

    def _raise_for_status(self, res: httpx.Response) -> None:
        if res.status_code < 400:
            return

        try:
            request = res.request
        except RuntimeError:
            method = "HTTP"
            url = ""
        else:
            method = request.method
            url = request.url
        raise SupabaseError(f"{method} {url} failed {res.status_code}: {res.text}")
