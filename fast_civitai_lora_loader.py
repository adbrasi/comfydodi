import json
import os
import shutil
import subprocess
from typing import Optional, Tuple

import requests

import folder_paths

from .utils import short_paths_map, model_path

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
HISTORY_FILE = os.path.join(ROOT_PATH, "download_history.json")
LORA_PATHS = folder_paths.folder_names_and_paths["loras"][0]

MSG_PREFIX = '\33[1m\33[34m[CivitAI] \33[0m'
WARN_PREFIX = '\33[1m\33[34m[CivitAI]\33[0m\33[93m Warning: \33[0m'
ERR_PREFIX = '\33[1m\33[31m[CivitAI]\33[0m\33[1m Error: \33[0m'


def _load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as history_file:
                return json.load(history_file)
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def _save_history(history: dict) -> None:
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as history_file:
            json.dump(history, history_file, indent=4, ensure_ascii=False)
    except OSError:
        pass


def _find_cached_entry(history: dict, model_id: int, version_id: Optional[int]) -> Optional[str]:
    model_entries = history.get(str(model_id), [])
    for entry in model_entries:
        if version_id is not None and entry.get("id") != version_id:
            continue
        for file_details in entry.get("files", []):
            name = file_details.get("name")
            if not name:
                continue
            resolved = model_path(name, LORA_PATHS)
            if resolved:
                return name
    return None


def _record_entry(history: dict, model_id: int, version_id: Optional[int], file_name: str, download_url: str) -> None:
    model_key = str(model_id)
    version_entry = None
    for entry in history.get(model_key, []):
        if entry.get("id") == version_id:
            version_entry = entry
            break
    if version_entry is None:
        version_entry = {"id": version_id, "files": []}
        history.setdefault(model_key, []).append(version_entry)
    files = version_entry.setdefault("files", [])
    for existing in files:
        if existing.get("name") == file_name:
            return
    files.append({
        "id": None,
        "name": file_name,
        "downloadUrl": download_url,
    })
    _save_history(history)


class FastCivitAIDownloader:
    api_root = "https://civitai.com/api/v1"

    def __init__(
        self,
        token: Optional[str],
        download_dir: str,
        timeout: int = 20,
        fallback: str = "aria2c",
        download_chunks: int = 16,
        prefer_tools_first: bool = True,
    ) -> None:
        token_value = token.strip() if isinstance(token, str) else token
        self.token = token_value if token_value else None
        self.download_dir = download_dir
        self.timeout = max(5, int(timeout) if timeout else 20)
        self.fallback = fallback or "aria2c"
        self.download_chunks = max(1, int(download_chunks) if download_chunks else 16)
        self.prefer_tools_first = prefer_tools_first
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "comfydodi-fast-lora/1.0"})
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.chunk_bytes = self.download_chunks * 1024 * 1024

    def _fetch(self, endpoint: str) -> dict:
        url = f"{self.api_root}/{endpoint.lstrip('/')}"
        params = {"token": self.token} if self.token else None
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status {response.status_code}")
        return response.json()

    def _resolve_download(self, model_id: int, version_id: Optional[int]) -> Tuple[int, Optional[int], str, str]:
        if version_id:
            try:
                details = self._fetch(f"model-versions/{version_id}")
            except RuntimeError as error:
                print(f"{WARN_PREFIX}Falling back to direct download for version {version_id}: {error}")
                return self._resolve_from_version_only(model_id, version_id)
        else:
            details = self._fetch(f"models/{model_id}")
            versions = details.get("modelVersions") or []
            if not versions:
                raise RuntimeError("Model has no versions available")
            details = versions[0]
            version_id = details.get("id")

        files = details.get("files") or []
        if not files:
            raise RuntimeError("Model version has no downloadable files")
        primary_file = next((item for item in files if item.get("primary")), files[0])
        download_url = primary_file.get("downloadUrl") or details.get("downloadUrl")
        if not download_url:
            raise RuntimeError("No download URL returned by CivitAI")
        file_name = primary_file.get("name") or os.path.basename(download_url)
        download_url = self._with_token(download_url)
        return model_id, version_id, file_name, download_url

    def _resolve_from_version_only(self, model_id: int, version_id: int) -> Tuple[int, int, str, str]:
        download_url = self._with_token(f"https://civitai.com/api/download/models/{version_id}")
        file_name = self._probe_filename(download_url)
        if not file_name:
            file_name = f"civitai_model_{version_id}.safetensors"
        return model_id, version_id, file_name, download_url

    def _probe_filename(self, url: str) -> Optional[str]:
        try:
            response = self.session.head(url, allow_redirects=True, timeout=self.timeout)
            if response.status_code >= 400:
                return None
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition:
                marker = "filename="
                if marker in content_disposition:
                    name = content_disposition.split(marker, 1)[1].strip('"')
                    if name:
                        return name
            return os.path.basename(response.url or url)
        except requests.exceptions.RequestException:
            return None

    def _with_token(self, url: str) -> str:
        if not self.token:
            return url
        if "token=" in url:
            return url
        separator = '&' if '?' in url else '?'
        return f"{url}{separator}token={self.token}"

    def download(self, model_id: int, version_id: Optional[int]) -> Tuple[str, str]:
        resolved_model_id, resolved_version_id, file_name, download_url = self._resolve_download(model_id, version_id)
        os.makedirs(self.download_dir, exist_ok=True)
        destination = os.path.join(self.download_dir, file_name)

        external_error = None
        if self.prefer_tools_first:
            try:
                self._download_with_external(download_url, destination)
                return file_name, download_url
            except RuntimeError as error:
                external_error = error
                print(f"{WARN_PREFIX}External downloader failed, retrying via API: {error}")

        try:
            self._download_with_requests(download_url, destination)
            return file_name, download_url
        except Exception as primary_error:
            print(f"{ERR_PREFIX}Direct download failed: {primary_error}")
            if not self.prefer_tools_first:
                self._download_with_external(download_url, destination)
                return file_name, download_url
            raise primary_error if external_error is None else RuntimeError(f"{external_error}; {primary_error}")

    def _download_with_requests(self, url: str, destination: str) -> None:
        try:
            with self.session.get(url, stream=True, timeout=self.timeout) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                with open(destination, "wb") as file:
                    for chunk in response.iter_content(chunk_size=self.chunk_bytes):
                        if chunk:
                            file.write(chunk)
        except Exception:
            if os.path.exists(destination):
                try:
                    os.remove(destination)
                except OSError:
                    pass
            raise

    def _download_with_external(self, url: str, destination: str) -> None:
        if os.path.exists(destination):
            try:
                os.remove(destination)
            except OSError:
                pass

        commands = self._external_commands(url, destination)
        if not commands:
            raise RuntimeError("No compatible external downloader found (install aria2c, wget, or curl)")

        error_messages = []
        for command in commands:
            try:
                completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if completed.returncode == 0 and os.path.exists(destination):
                    return
            except (subprocess.CalledProcessError, FileNotFoundError) as error:
                print(f"{ERR_PREFIX}Fallback `{command[0]}` failed: {error}")
                if os.path.exists(destination):
                    try:
                        os.remove(destination)
                    except OSError:
                        pass
                error_messages.append(str(error))
                continue

        if error_messages:
            raise RuntimeError("; ".join(error_messages))
        raise RuntimeError("External download attempts failed")

    def _external_commands(self, url: str, destination: str):
        commands = []
        if self.fallback == "requests_only":
            return commands

        directory = os.path.dirname(destination)
        filename = os.path.basename(destination)
        fallback_order = []
        if self.fallback == "auto" or self.fallback == "aria2c":
            fallback_order.append("aria2c")
        if self.fallback == "auto" or self.fallback == "wget":
            fallback_order.append("wget")
        if self.fallback == "auto" or self.fallback == "curl":
            fallback_order.append("curl")
        if self.fallback not in ("auto", "aria2c", "wget", "curl", "requests_only"):
            fallback_order = [self.fallback]

        for tool in fallback_order:
            if shutil.which(tool) is None:
                continue
            if tool == "aria2c":
                command = [
                    "aria2c",
                    "--allow-overwrite=true",
                    "--auto-file-renaming=false",
                    "-x", str(self.download_chunks),
                    "-s", str(self.download_chunks),
                    "-d", directory,
                    "-o", filename,
                ]
                if self.token:
                    command.extend(["--header", f"Authorization: Bearer {self.token}"])
                command.append(url)
                commands.append(command)
            elif tool == "wget":
                command = [
                    "wget",
                    "-O", destination,
                    "--content-disposition",
                ]
                if self.token:
                    command.extend(["--header", f"Authorization: Bearer {self.token}"])
                command.append(url)
                commands.append(command)
            elif tool == "curl":
                command = [
                    "curl",
                    "-L",
                    url,
                    "-o", destination,
                ]
                if self.token:
                    command.extend(["-H", f"Authorization: Bearer {self.token}"])
                commands.append(command)

        return commands


def _resolve_download_path(download_path_key: Optional[str]) -> str:
    path_map = short_paths_map(LORA_PATHS)
    if download_path_key and download_path_key in path_map:
        return path_map[download_path_key]
    return LORA_PATHS[0]


def _parse_lora_air(lora_air: str) -> Tuple[int, Optional[int]]:
    if not lora_air:
        raise ValueError("LORA AIR identifier required (ex: 12345@67890)")
    parts = lora_air.strip().split("@", 1)
    try:
        model_id = int(parts[0])
    except ValueError as exc:
        raise ValueError("Invalid model ID provided") from exc
    version_id = None
    if len(parts) > 1 and parts[1]:
        try:
            version_id = int(parts[1])
        except ValueError as exc:
            raise ValueError("Invalid version ID provided") from exc
    return model_id, version_id


class CivitAI_Fast_LORA_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        loras.insert(0, 'none')
        lora_paths = short_paths_map(LORA_PATHS)
        return {
            "required": {
                "lora_air": ("STRING", {"default": "{model_id}@{model_version}", "multiline": False}),
            },
            "optional": {
                "lora_name": (loras,),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "download_path": (list(lora_paths),),
                "download_chunks": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "timeout_seconds": ("INT", {"default": 20, "min": 5, "max": 300, "step": 5}),
                "fallback_downloader": (["aria2c", "auto", "wget", "curl", "requests_only"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_fast_lora"
    CATEGORY = "CivitAI/Fast"

    def load_fast_lora(
        self,
        lora_air: str,
        lora_name: str = 'none',
        api_key: str = '',
        download_path: Optional[str] = None,
        download_chunks: int = 16,
        timeout_seconds: int = 20,
        fallback_downloader: str = "aria2c",
    ):
        if lora_name and lora_name != 'none':
            print(f"{MSG_PREFIX}Using existing LORA: {lora_name}")
            return (lora_name,)

        model_id, version_id = _parse_lora_air(lora_air)
        history = _load_history()
        cached_name = _find_cached_entry(history, model_id, version_id)
        if cached_name:
            print(f"{MSG_PREFIX}Found cached LORA `{cached_name}` for {model_id}@{version_id or 'latest'}")
            return (cached_name,)

        resolved_path = _resolve_download_path(download_path)
        token = api_key or os.environ.get("CIVITAI_API_TOKEN")
        prefer_tools_first = fallback_downloader != "requests_only"
        downloader = FastCivitAIDownloader(
            token=token,
            download_dir=resolved_path,
            timeout=timeout_seconds,
            fallback=fallback_downloader,
            download_chunks=download_chunks,
            prefer_tools_first=prefer_tools_first,
        )
        file_name, download_url = downloader.download(model_id, version_id)
        _record_entry(history, model_id, version_id, file_name, download_url)
        print(f"{MSG_PREFIX}Fast downloaded `{file_name}` to `{resolved_path}`")
        return (file_name,)
