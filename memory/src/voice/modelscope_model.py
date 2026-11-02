"""Resolve FunASR model files before handing a local directory to AutoModel."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml


DOMESTIC_ENDPOINT = "https://modelscope.cn"
INTERNATIONAL_ENDPOINT = "https://www.modelscope.ai"


def _metadata_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [path for item in value.values() for path in _metadata_paths(item)]
    if isinstance(value, list):
        return [path for item in value for path in _metadata_paths(item)]
    return []


def _validate_snapshot(directory: Path) -> None:
    """ModelScope's offline lookup also returns partially downloaded snapshots."""
    metadata_file = directory / "configuration.json"
    if metadata_file.is_file():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid FunASR snapshot metadata: {metadata_file}")
        paths = metadata.get("file_path_metas", {})
        if not isinstance(paths, dict):
            raise ValueError(f"Invalid FunASR snapshot file mapping: {metadata_file}")
        config_name = paths.get("config")
        weights_name = paths.get("init_param")
        required = _metadata_paths(paths)
    else:
        config_name, weights_name = "config.yaml", "model.pt"
        required = [config_name, weights_name]
    if not isinstance(config_name, str) or not isinstance(weights_name, str):
        raise ValueError(f"FunASR snapshot has no config/weights mapping: {directory}")
    for name in required:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"FunASR snapshot is incomplete; missing or empty file: {path}")
    config = yaml.safe_load((directory / config_name).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("model"), str):
        raise ValueError(f"FunASR snapshot has no model class: {directory / config_name}")


def resolve_modelscope_model_path(
    model_id: str,
    *,
    revision: str = "master",
    logger: logging.Logger | None = None,
) -> str:
    """Prefer complete cached files, then try the domestic and international hubs.

    Keep hub errors outside FunASR: its downloader prints the original error
    and continues with the unresolved repository ID as if it were a model class.
    No process-wide endpoint or downloader is changed here.
    """
    logger = logger or logging.getLogger(__name__)
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        _validate_snapshot(local_path)
        return str(local_path.resolve())

    from modelscope.hub.snapshot_download import snapshot_download

    # Retain FunASR's documented speaker-model alias without changing hubs for
    # the separate HuggingFace-based Paraformer backends.
    repository = (
        "iic/speech_campplus_sv_zh-cn_16k-common" if model_id == "cam++" else model_id
    )
    try:
        cached = Path(snapshot_download(
            repository, revision=revision, local_files_only=True,
        ))
        _validate_snapshot(cached)
    except Exception as error:
        # SDK cache exceptions are not stable across ModelScope versions.
        logger.info("FunASR model cache unavailable model_id=%s reason=%s", repository, error)
    else:
        logger.info("FunASR model cache hit model_id=%s path=%s", repository, cached)
        return str(cached.resolve())

    errors = []
    for endpoint in (DOMESTIC_ENDPOINT, INTERNATIONAL_ENDPOINT):
        try:
            downloaded = Path(snapshot_download(
                repository, revision=revision, endpoint=endpoint,
            ))
            _validate_snapshot(downloaded)
        except Exception as error:
            errors.append(f"{endpoint}: {error}")
            logger.warning(
                "FunASR model download failed model_id=%s endpoint=%s; %s",
                repository, endpoint,
                "switching to international hub" if endpoint == DOMESTIC_ENDPOINT else "no hubs left",
                exc_info=True,
            )
            if endpoint == INTERNATIONAL_ENDPOINT:
                raise RuntimeError(
                    f"Cannot download FunASR model {repository}: " + "; ".join(errors)
                ) from error
        else:
            logger.info(
                "FunASR model download ready model_id=%s endpoint=%s path=%s",
                repository, endpoint, downloaded,
            )
            return str(downloaded.resolve())
    raise AssertionError("model hub attempts exhausted without a result")
