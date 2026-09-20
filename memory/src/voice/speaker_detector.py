import sys
import tempfile
import json
import os
import numpy as np
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



def _l2_normalize(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector if norm <= 0.0 else vector / norm


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_vector = _l2_normalize(left)
    right_vector = _l2_normalize(right)
    if left_vector.size == 0 or right_vector.size == 0:
        return 0.0
    keep = min(left_vector.size, right_vector.size)
    return float(np.dot(left_vector[:keep], right_vector[:keep]))

DEFAULT_USER_REFERENCE_AUDIO_DIR = (
    Path(__file__).resolve().parents[2] / "assets/user_reference_audio_files"
)
DEFAULT_REFERENCE_EMBEDDINGS_PATH = (
    Path(__file__).resolve().parents[2] / "tmp/mcp/speaker_reference_embeddings.npz"
)
SUPPORTED_REFERENCE_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def _find_embedding_array(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.size > 8:
            return value
        return None
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return _find_embedding_array(array)
    if isinstance(value, dict):
        preferred_keys = ("spk_embedding", "speaker_embedding", "embedding", "embeddings")
        for key in preferred_keys:
            if key in value:
                found = _find_embedding_array(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = _find_embedding_array(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_embedding_array(item)
            if found is not None:
                return found
    return None

@dataclass
class SpeakerReference:
    speaker_id: int
    embeddings: List[np.ndarray]


@dataclass
class SpeakerAssignment:
    speaker_id: int
    similarity: Optional[float]
    profile_count: int
    reference_count: int
    reason: str


@dataclass
class SpeakerVerificationResult:
    is_user: bool
    similarity: Optional[float]
    threshold: float
    reference_count: int
    reason: str


class SpeakerReferenceDetector:
    def __init__(
        self,
        sample_rate: int,
        device: Optional[str],
        model_id: str,
        similarity_threshold: float,
        reference_memory_size: int,
        min_new_reference_segments: int = 2,
        min_new_reference_duration_s: float = 3.0,
        min_reference_segment_duration_s: float = 1.0,
        user_similarity_threshold: Optional[float] = None,
        user_reference_audio_dir: Optional[Path] = None,
        reference_embeddings_path: Optional[Path] = None,
        speaker_assignment_method: str = "reference",
        embedding_batch_size: int = 16,
    ) -> None:
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "CAMPPlus speaker identification requires FunASR. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        kwargs = {"model": model_id}
        if device and device != "cpu":
            kwargs["device"] = device
        self.model = AutoModel(**kwargs)
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.reference_memory_size = max(1, reference_memory_size)
        self.min_new_reference_segments = max(1, min_new_reference_segments)
        self.min_new_reference_duration_s = max(0.0, min_new_reference_duration_s)
        self.min_reference_segment_duration_s = max(
            0.0,
            min_reference_segment_duration_s,
        )
        if speaker_assignment_method not in {"reference", "agglomerative"}:
            raise ValueError("speaker_assignment_method must be reference or agglomerative.")
        self.speaker_assignment_method = speaker_assignment_method
        self.embedding_batch_size = max(1, int(embedding_batch_size))
        self.user_similarity_threshold = (
            similarity_threshold
            if user_similarity_threshold is None
            else user_similarity_threshold
        )
        self.reference_embeddings: List[SpeakerReference] = []
        self.user_reference_embeddings: List[np.ndarray] = []
        self.user_reference_audio_dir = (
            user_reference_audio_dir.expanduser().resolve()
            if user_reference_audio_dir is not None
            else DEFAULT_USER_REFERENCE_AUDIO_DIR
        )
        self.reference_embeddings_path = (
            reference_embeddings_path.expanduser().resolve()
            if reference_embeddings_path is not None
            else DEFAULT_REFERENCE_EMBEDDINGS_PATH
        )
        self.next_speaker_id = 1
        self._load_reference_embeddings()
        self.reload_user_reference_audios()

    def load_user_reference_audios(self, audio_paths: List[Path]) -> None:
        self.user_reference_embeddings = [
            self._encode_audio(self._load_reference_audio(path))
            for path in audio_paths
        ]

    def reload_user_reference_audios(self, extra_audio_paths: Optional[List[Path]] = None) -> int:
        audio_paths = self._user_reference_audio_files()
        if extra_audio_paths:
            audio_paths.extend(path.expanduser().resolve() for path in extra_audio_paths)
        self.load_user_reference_audios(audio_paths)
        return len(self.user_reference_embeddings)

    def judge_user_speaker_embedding(self, audio: np.ndarray) -> SpeakerVerificationResult:
        if not self.user_reference_embeddings:
            return SpeakerVerificationResult(
                is_user=True,
                similarity=None,
                threshold=self.user_similarity_threshold,
                reference_count=0,
                reason="no_user_reference",
            )
        audio = self._segment_audio(audio)
        if audio.size == 0:
            return SpeakerVerificationResult(
                is_user=False,
                similarity=None,
                threshold=self.user_similarity_threshold,
                reference_count=len(self.user_reference_embeddings),
                reason="empty_audio",
            )
        embedding = self._encode_audio(audio)
        similarities = [
            _cosine_similarity(reference_embedding, embedding)
            for reference_embedding in self.user_reference_embeddings
        ]
        best_similarity = max(similarities)
        is_user = best_similarity >= self.user_similarity_threshold
        return SpeakerVerificationResult(
            is_user=is_user,
            similarity=round(best_similarity, 4),
            threshold=self.user_similarity_threshold,
            reference_count=len(self.user_reference_embeddings),
            reason="matched_user" if is_user else "rejected_non_user",
        )

    def identify_batch_segments_speaker(self, audio_segments: List[np.ndarray]) -> List[SpeakerAssignment]:
        if self.speaker_assignment_method == "agglomerative":
            return self.identify_speaker_using_agglomerative_clustering(audio_segments)
        return self.identify_speaker_using_comparing_reference_embeddings(audio_segments)

    def identify_speaker_using_comparing_reference_embeddings(self, audio_segments: List[np.ndarray]) -> List[SpeakerAssignment]:
        all_speaker_assignments = []

        for audio_segment in audio_segments:
            embedding = self._encode_audio(self._segment_audio(audio_segment))
            reference_eligible = self._reference_segment_is_eligible(
                self._segment_duration_s(audio_segment)
            )
            if not self.reference_embeddings:
                if not reference_eligible:
                    all_speaker_assignments.append(
                        SpeakerAssignment(
                            speaker_id=-1,
                            similarity=None,
                            profile_count=0,
                            reference_count=0,
                            reason="short_segment_unknown_reference",
                        )
                    )
                    continue
                speaker_id = self._create_reference(embedding)
                all_speaker_assignments.append(
                    SpeakerAssignment(
                        speaker_id=speaker_id,
                        similarity=None,
                        profile_count=len(self.reference_embeddings),
                        reference_count=1,
                        reason="new_reference",
                    )
                )
                continue

            similarities = [
                _cosine_similarity(self._reference_embedding(reference), embedding)
                for reference in self.reference_embeddings
            ]
            best_index = int(np.argmax(similarities))
            best_similarity = similarities[best_index]
            if best_similarity >= self.similarity_threshold:
                reference = self.reference_embeddings[best_index]
                if reference_eligible:
                    self._append_reference_embedding(reference, embedding)
                all_speaker_assignments.append(
                    SpeakerAssignment(
                        speaker_id=reference.speaker_id,
                        similarity=round(best_similarity, 4),
                        profile_count=len(self.reference_embeddings),
                        reference_count=len(reference.embeddings),
                        reason=(
                            "matched_reference"
                            if reference_eligible
                            else "matched_reference_short_segment"
                        ),
                    )
                )
                continue

            if not reference_eligible:
                all_speaker_assignments.append(
                    SpeakerAssignment(
                        speaker_id=-1,
                        similarity=round(best_similarity, 4),
                        profile_count=len(self.reference_embeddings),
                        reference_count=0,
                        reason="short_segment_unknown_reference",
                    )
                )
                continue

            speaker_id = self._create_reference(embedding)
            all_speaker_assignments.append(
                SpeakerAssignment(
                    speaker_id=speaker_id,
                    similarity=round(best_similarity, 4),
                    profile_count=len(self.reference_embeddings),
                    reference_count=1,
                    reason="new_reference",
                )
            )
        self._save_reference_embeddings()
        return all_speaker_assignments

    def identify_speaker_using_agglomerative_clustering(
        self,
        audio_segments: List[np.ndarray],
    ) -> List[SpeakerAssignment]:
        if not audio_segments:
            return []

        embeddings = [
            self._encode_audio(self._segment_audio(audio_segment))
            for audio_segment in audio_segments
        ]
        labels = self._cluster_embeddings(np.vstack(embeddings))
        assignments: List[Optional[SpeakerAssignment]] = [None] * len(audio_segments)

        for label in self._ordered_cluster_labels(labels):
            cluster_indexes = [index for index, item_label in enumerate(labels) if item_label == label]
            cluster_segments = [audio_segments[index] for index in cluster_indexes]
            cluster_segment_embeddings = [embeddings[index] for index in cluster_indexes]
            cluster_mean_embedding = _l2_normalize(np.mean(np.vstack(cluster_segment_embeddings), axis=0))
            user_assignment = self._assign_cluster_to_user_reference(cluster_mean_embedding)
            if user_assignment is not None:
                assignment = user_assignment
            else:
                assignment = self._assign_cluster_to_reference(
                    cluster_segment_embeddings=cluster_segment_embeddings,
                    cluster_segment_durations=[
                        self._segment_duration_s(segment)
                        for segment in cluster_segments
                    ],
                )
            for index in cluster_indexes:
                assignments[index] = assignment

        self._save_reference_embeddings()
        return [assignment for assignment in assignments if assignment is not None]

    def _assign_cluster_to_user_reference(
        self,
        cluster_mean_embedding: np.ndarray,
    ) -> Optional[SpeakerAssignment]:
        if not self.user_reference_embeddings:
            return None
        similarities = [
            _cosine_similarity(reference_embedding, cluster_mean_embedding)
            for reference_embedding in self.user_reference_embeddings
        ]
        best_similarity = max(similarities)
        if best_similarity < self.user_similarity_threshold:
            return None
        return SpeakerAssignment(
            speaker_id=0,
            similarity=round(best_similarity, 4),
            profile_count=len(self.reference_embeddings),
            reference_count=len(self.user_reference_embeddings),
            reason="user_reference_matched_user",
        )

    def _assign_cluster_to_reference(
        self,
        cluster_segment_embeddings: List[np.ndarray],
        cluster_segment_durations: List[float],
    ) -> SpeakerAssignment:
        # Short segments still participate in the current cluster decision,
        # but they must not update the long-term speaker profile.
        eligible_embeddings = [
            embedding
            for embedding, segment_duration_s in zip(
                cluster_segment_embeddings,
                cluster_segment_durations,
            )
            if self._reference_segment_is_eligible(segment_duration_s)
        ]
        eligible_duration_s = sum(
            segment_duration_s
            for segment_duration_s in cluster_segment_durations
            if self._reference_segment_is_eligible(segment_duration_s)
        )
        can_create_reference = self._can_create_reference(
            len(eligible_embeddings),
            eligible_duration_s,
        )
        if not eligible_embeddings:
            return SpeakerAssignment(
                speaker_id=-1,
                similarity=None,
                profile_count=len(self.reference_embeddings),
                reference_count=0,
                reason="short_cluster_unknown_reference",
            )
        eligible_cluster_mean_embedding = _l2_normalize(
            np.mean(np.vstack(eligible_embeddings), axis=0)
        )
        if not self.reference_embeddings:
            if not can_create_reference:
                return SpeakerAssignment(
                    speaker_id=-1,
                    similarity=None,
                    profile_count=0,
                    reference_count=0,
                    reason="cluster_insufficient_evidence_unknown",
                )

            speaker_id = self._create_reference(eligible_embeddings[0])
            reference = self.reference_embeddings[-1]
            for embedding in eligible_embeddings[1:]:
                self._append_reference_embedding(reference, embedding)
            return SpeakerAssignment(
                speaker_id=speaker_id,
                similarity=None,
                profile_count=len(self.reference_embeddings),
                reference_count=len(reference.embeddings),
                reason="cluster_new_reference",
            )

        similarities = [
            _cosine_similarity(
                self._reference_embedding(reference),
                eligible_cluster_mean_embedding,
            )
            for reference in self.reference_embeddings
        ]
        best_index = int(np.argmax(similarities))
        best_similarity = similarities[best_index]
        if best_similarity >= self.similarity_threshold:
            reference = self.reference_embeddings[best_index]
            for embedding in eligible_embeddings:
                self._append_reference_embedding(reference, embedding)
            return SpeakerAssignment(
                speaker_id=reference.speaker_id,
                similarity=round(best_similarity, 4),
                profile_count=len(self.reference_embeddings),
                reference_count=len(reference.embeddings),
                reason="cluster_matched_reference",
            )

        if not can_create_reference:
            reference = self.reference_embeddings[best_index]
            return SpeakerAssignment(
                speaker_id=reference.speaker_id,
                similarity=round(best_similarity, 4),
                profile_count=len(self.reference_embeddings),
                reference_count=len(reference.embeddings),
                reason="cluster_weak_matched_reference",
            )

        speaker_id = self._create_reference(eligible_embeddings[0])
        reference = self.reference_embeddings[-1]
        for embedding in eligible_embeddings[1:]:
            self._append_reference_embedding(reference, embedding)
        return SpeakerAssignment(
            speaker_id=speaker_id,
            similarity=round(best_similarity, 4),
            profile_count=len(self.reference_embeddings),
            reference_count=len(reference.embeddings),
            reason="cluster_new_reference",
        )

    def _cluster_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 1:
            return np.zeros(1, dtype=np.int32)

        distance_threshold = max(0.0, min(2.0, 1.0 - self.similarity_threshold))
        try:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=distance_threshold,
            )
        except TypeError:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                affinity="cosine",
                linkage="average",
                distance_threshold=distance_threshold,
            )
        return clustering.fit_predict(embeddings).astype(np.int32)

    @staticmethod
    def _ordered_cluster_labels(labels: np.ndarray) -> List[int]:
        return sorted(set(int(label) for label in labels), key=lambda label: int(np.where(labels == label)[0][0]))

    @staticmethod
    def _segment_audio(audio_segment: Any) -> np.ndarray:
        if hasattr(audio_segment, "audio"):
            return np.asarray(audio_segment.audio, dtype=np.float32)
        return np.asarray(audio_segment, dtype=np.float32)

    def _segment_duration_s(self, audio_segment: Any) -> float:
        if hasattr(audio_segment, "duration"):
            return float(audio_segment.duration)
        return len(self._segment_audio(audio_segment)) / self.sample_rate

    def _user_reference_audio_files(self) -> List[Path]:
        self.user_reference_audio_dir.mkdir(parents=True, exist_ok=True)
        return sorted(
            path
            for path in self.user_reference_audio_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_REFERENCE_AUDIO_EXTENSIONS
        )

    def _can_create_reference(self, segment_count: int, duration_s: float) -> bool:
        return (
            segment_count >= self.min_new_reference_segments
            or duration_s >= self.min_new_reference_duration_s
        )

    def _reference_segment_is_eligible(self, duration_s: float) -> bool:
        """Return whether a segment may update the long-term profile."""
        return duration_s >= self.min_reference_segment_duration_s

    def _create_reference(self, embedding: np.ndarray) -> int:
        speaker_id = self.next_speaker_id
        self.next_speaker_id += 1
        self.reference_embeddings.append(
            SpeakerReference(
                speaker_id=speaker_id,
                embeddings=[_l2_normalize(embedding)],
            )
        )
        return speaker_id

    def _append_reference_embedding(self, reference: SpeakerReference, embedding: np.ndarray) -> None:
        reference.embeddings.append(_l2_normalize(embedding))
        if len(reference.embeddings) > self.reference_memory_size:
            reference.embeddings = reference.embeddings[-self.reference_memory_size :]

    def _load_reference_embeddings(self) -> None:
        path = self.reference_embeddings_path
        if not path.is_file():
            return

        try:
            with np.load(path, allow_pickle=False) as archive:
                if "metadata" not in archive.files:
                    raise ValueError("missing metadata")
                metadata_value = archive["metadata"]
                metadata = json.loads(str(metadata_value.item()))
                if metadata.get("schema_version") != 1:
                    raise ValueError(
                        f"unsupported schema_version={metadata.get('schema_version')}"
                    )
                if int(metadata.get("sample_rate", -1)) != self.sample_rate:
                    raise ValueError(
                        f"sample_rate mismatch: saved={metadata.get('sample_rate')} "
                        f"current={self.sample_rate}"
                    )
                if str(metadata.get("model_id")) != str(self.model_id):
                    raise ValueError(
                        f"model_id mismatch: saved={metadata.get('model_id')} "
                        f"current={self.model_id}"
                    )

                references: List[SpeakerReference] = []
                expected_dim = int(metadata.get("embedding_dim", 0))
                for item in metadata.get("speakers", []):
                    speaker_id = int(item["speaker_id"])
                    key = str(item["key"])
                    values = np.asarray(archive[key], dtype=np.float32)
                    if values.ndim != 2 or values.shape[0] == 0:
                        raise ValueError(f"invalid embedding matrix for speaker {speaker_id}")
                    if expected_dim and values.shape[1] != expected_dim:
                        raise ValueError(
                            f"embedding dimension mismatch for speaker {speaker_id}: "
                            f"saved={values.shape[1]} expected={expected_dim}"
                        )
                    references.append(
                        SpeakerReference(
                            speaker_id=speaker_id,
                            embeddings=[
                                _l2_normalize(row)
                                for row in values[-self.reference_memory_size :]
                            ],
                        )
                    )

            self.reference_embeddings = sorted(
                references,
                key=lambda reference: reference.speaker_id,
            )
            self.next_speaker_id = max(
                (reference.speaker_id for reference in self.reference_embeddings),
                default=0,
            ) + 1
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.reference_embeddings = []
            self.next_speaker_id = 1
            # A stale or incompatible profile must not poison the current run.
            # It will be replaced after the first successful identification.
            self._reference_load_error = str(exc)

    def _save_reference_embeddings(self) -> None:
        if not self.reference_embeddings:
            return

        path = self.reference_embeddings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        speakers = []
        embedding_dim = 0
        for reference in self.reference_embeddings:
            key = f"speaker_{reference.speaker_id}"
            values = np.asarray(reference.embeddings, dtype=np.float32)
            if values.ndim != 2 or values.shape[0] == 0:
                continue
            embedding_dim = values.shape[1]
            arrays[key] = values[-self.reference_memory_size :]
            speakers.append({"speaker_id": reference.speaker_id, "key": key})
        if not arrays:
            return

        metadata = {
            "schema_version": 1,
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "embedding_dim": embedding_dim,
            "reference_memory_size": self.reference_memory_size,
            "speakers": speakers,
        }
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npz",
                prefix=f".{path.name}.",
                dir=path.parent,
                delete=False,
            ) as temp_file:
                temporary_path = Path(temp_file.name)
            np.savez_compressed(
                temporary_path,
                metadata=np.asarray(json.dumps(metadata), dtype=np.str_),
                **arrays,
            )
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _reference_embedding(reference: SpeakerReference) -> np.ndarray:
        return _l2_normalize(np.mean(reference.embeddings, axis=0))

    def _encode_audio_batch(self, audios: List[np.ndarray]) -> List[np.ndarray]:
        """Encode multiple audio arrays in one CAM++ inference call."""
        if not audios:
            return []

        normalized_audios = [
            np.asarray(audio, dtype=np.float32).reshape(-1) for audio in audios
        ]
        valid_indexes = [
            index for index, audio in enumerate(normalized_audios) if audio.size
        ]
        embeddings: List[Optional[np.ndarray]] = [None] * len(normalized_audios)
        if valid_indexes:
            valid_audios = [normalized_audios[index] for index in valid_indexes]
            outputs = self.model.generate(
                input=valid_audios,
                batch_size=len(valid_audios),
            )
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            batch_embeddings: List[np.ndarray] = []
            if len(outputs) == 1 and len(valid_audios) > 1:
                embedding = _find_embedding_array(outputs[0])
                if embedding is not None:
                    embedding_array = np.asarray(embedding, dtype=np.float32)
                    if embedding_array.ndim >= 2 and embedding_array.shape[0] == len(
                        valid_audios
                    ):
                        batch_embeddings = [
                            embedding_array[index] for index in range(len(valid_audios))
                        ]
            if not batch_embeddings:
                if len(outputs) != len(valid_audios):
                    raise RuntimeError(
                        "CAMPPlus batch embedding count does not match audio count: "
                        f"outputs={len(outputs)} audios={len(valid_audios)}"
                    )
                for output in outputs:
                    embedding = _find_embedding_array(output)
                    if embedding is None:
                        raise RuntimeError(
                            "Could not find a speaker embedding in CAMPPlus output. "
                            f"Output type={type(output)!r}, value={output!r}"
                        )
                    embedding_array = np.asarray(embedding, dtype=np.float32)
                    if embedding_array.ndim >= 2 and embedding_array.shape[0] == 1:
                        embedding_array = embedding_array[0]
                    batch_embeddings.append(embedding_array)
            for index, embedding in zip(valid_indexes, batch_embeddings):
                embeddings[index] = _l2_normalize(embedding)

        return [
            embedding
            if embedding is not None
            else np.zeros(192, dtype=np.float32)
            for embedding in embeddings
        ]

    def _encode_audio(self, audio: np.ndarray) -> np.ndarray:
        return self._encode_audio_batch([audio])[0]

    def _load_reference_audio(self, path: Path) -> np.ndarray:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Reference speaker audio must be {self.sample_rate} Hz, "
                f"got {sample_rate} Hz: {path}"
            )
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return np.asarray(audio, dtype=np.float32)
