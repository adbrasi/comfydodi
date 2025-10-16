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

    def __init__(self, token: Optional[str], download_dir: str, timeout: int = 20, fallback: str = "auto") -> None:
        self.token = token.strip() if isinstance(token, str) else token
        self.download_dir = download_dir
        self.timeout = max(5, int(timeout) if timeout else 20)
        self.fallback = fallback or "auto"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "comfydodi-fast-lora/1.0"})
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _fetch(self, endpoint: str) -> dict:
        url = f"{self.api_root}/{endpoint.lstrip('/')}"
        params = {"token": self.token} if self.token else None
        response = self.session.get(url, params=params, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(f"API request failed with status {response.status_code}")
        return response.json()

    def _resolve_download(self, model_id: int, version_id: Optional[int]) -> Tuple[int, int, str, str]:
        if version_id:
            details = self._fetch(f"model-versions/{version_id}")
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
        primary_file = next((f for f in files if f.get("primary")), files[0])
        download_url = primary_file.get("downloadUrl") or details.get("downloadUrl")
        if not download_url:
            raise RuntimeError("No download URL returned by CivitAI")
        file_name = primary_file.get("name") or os.path.basename(download_url)
        download_url = self._with_token(download_url)
        return model_id, version_id or details.get("id"), file_name, download_url

    def _with_token(self, url: str) -> str:
        if not self.token:
            return url
        if "token=" in url:
            return url
        separator = '&' if '?' in url else '?'
        return f"{url}{separator}token={self.token}"

    def download(self, model_id: int, version_id: Optional[int]) -> Tuple[str, str]:
        _, _, file_name, download_url = self._resolve_download(model_id, version_id)
        os.makedirs(self.download_dir, exist_ok=True)
        destination = os.path.join(self.download_dir, file_name)
        try:
            self._download_with_requests(download_url, destination)
        except Exception as primary_error:
            print(f"{ERR_PREFIX}Fast download via API failed: {primary_error}")
            self._download_with_fallback(download_url, destination)
        return file_name, download_url

    def _download_with_requests(self, url: str, destination: str) -> None:
        try:
            with self.session.get(url, stream=True, timeout=self.timeout) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"HTTP {response.status_code}")
                with open(destination, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            file.write(chunk)
        except Exception:
            if os.path.exists(destination):
                try:
                    os.remove(destination)
                except OSError:
                    pass
            raise

    def _download_with_fallback(self, url: str, destination: str) -> None:
        error_messages = []
        if os.path.exists(destination):
            try:
                os.remove(destination)
            except OSError:
                pass
        for command in self._fallback_commands(url, destination):
            try:
                completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if completed.returncode == 0:
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
        raise RuntimeError("; ".join(error_messages) or "No fallback downloader available")

    def _fallback_commands(self, url: str, destination: str):
        directory = os.path.dirname(destination)
        filename = os.path.basename(destination)
        fallback_order = []
        if self.fallback == "auto":
            fallback_order = ["aria2c", "wget", "curl"]
        elif self.fallback == "requests_only":
            return []
        else:
            fallback_order = [self.fallback]

        for tool in fallback_order:
            if shutil.which(tool) is None:
                continue
            if tool == "aria2c":
                command = [
                    "aria2c",
                    "--allow-overwrite=true",
                    "--auto-file-renaming=false",
                    "-x", "16",
                    "-s", "16",
                    "-d", directory,
                    "-o", filename,
                ]
                if self.token:
                    command.extend(["--header", f"Authorization: Bearer {self.token}"])
                command.append(url)
                yield command
            elif tool == "wget":
                command = [
                    "wget",
                    "-O", destination,
                    "--content-disposition",
                ]
                if self.token:
                    command.extend(["--header", f"Authorization: Bearer {self.token}"])
                command.append(url)
                yield command
            elif tool == "curl":
                command = [
                    "curl",
                    "-L",
                    url,
                    "-o", destination,
                ]
                if self.token:
                    command.extend(["-H", f"Authorization: Bearer {self.token}"])
                yield command


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
                "fallback_downloader": (["auto", "aria2c", "wget", "curl", "requests_only"],),
                "timeout_seconds": ("INT", {"default": 20, "min": 5, "max": 300, "step": 5}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_fast_lora"
    CATEGORY = "CivitAI/Fast"

    def load_fast_lora(self, lora_air: str, lora_name: str = 'none', api_key: str = '', download_path: Optional[str] = None, fallback_downloader: str = "auto", timeout_seconds: int = 20):
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
        downloader = FastCivitAIDownloader(token=token, download_dir=resolved_path, timeout=timeout_seconds, fallback=fallback_downloader)
        file_name, download_url = downloader.download(model_id, version_id)
        _record_entry(history, model_id, version_id, file_name, download_url)
        print(f"{MSG_PREFIX}Fast downloaded `{file_name}` to `{resolved_path}`")
        return (file_name,)
