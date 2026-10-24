import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

CORE_CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]

ALL_CLASSES = CORE_CLASSES + ["silence", "unknown"]

LABEL2ID = {label: idx for idx, label in enumerate(ALL_CLASSES)}
ID2LABEL = {idx: label for idx, label in enumerate(ALL_CLASSES)}

SAMPLE_LENGTH = 16_000


class SpeechCommandsDataset(Dataset):
    def __init__(
        self,
        audio_dir: Path,
        split: str,
        validation_files: set[str],
        test_files: set[str],
        anomalies: set[str] | None = None,
        max_samples: int | None = None,
        add_silence: bool = True,
        skip_anomalies: bool = False,
        dataset_format: str = "auto",
    ):
        self.audio_dir = Path(audio_dir)
        self.split = split
        self.validation_files = validation_files
        self.test_files = test_files
        self.anomalies = anomalies or set()
        self.max_samples = max_samples
        self.skip_anomalies = skip_anomalies

        if dataset_format not in {"auto", "speech_commands", "kaggle_test"}:
            raise ValueError(
                f"Unknown dataset format: {dataset_format}. "
                "Expected 'auto', 'speech_commands' or 'kaggle_test'."
            )

        # Automatically detect the dataset structure.
        if dataset_format == "auto":
            self.dataset_format = self._detect_dataset_format()
        else:
            self.dataset_format = dataset_format

        # Kaggle test datasets contain audio files directly in audio_dir
        # and have no class labels.
        self.is_unlabeled = self.dataset_format == "kaggle_test"

        self.samples: list[dict[str, Any]] = []
        self.background_noises = (
            self._load_background_noises() if not self.is_unlabeled else []
        )

        self._build_samples(add_silence=add_silence)

    def _detect_dataset_format(self) -> str:
        """
        Detect the dataset structure.

        Speech Commands:
            audio/yes/*.wav
            audio/no/*.wav
            audio/_background_noise_/*.wav

        Kaggle test:
            audio/*.wav
        """
        direct_audio_files = list(self.audio_dir.glob("*.wav"))

        class_directories = [
            path
            for path in self.audio_dir.iterdir()
            if path.is_dir() and path.name != "_background_noise_"
        ]

        if direct_audio_files and not class_directories:
            return "kaggle_test"

        return "speech_commands"

    def _load_background_noises(self) -> list[np.ndarray]:
        noise_dir = self.audio_dir / "_background_noise_"

        if not noise_dir.exists():
            return []

        noises = []

        for path in noise_dir.glob("*.wav"):
            noise, _ = sf.read(path)

            if noise.ndim > 1:
                noise = noise.mean(axis=1)

            noises.append(noise.astype(np.float32))

        return noises

    def _build_samples(self, add_silence: bool) -> None:
        # Unlabeled Kaggle test split.
        # Files are stored directly in audio_dir, without class folders.
        if self.is_unlabeled:
            if self.split not in {"test", "all"}:
                raise ValueError(
                    f"The detected Kaggle test dataset has no labels. "
                    f"Split '{self.split}' is not supported. "
                    "Use split='test' or split='all'."
                )

            audio_files = sorted(self.audio_dir.glob("*.wav"))

            if self.max_samples is not None:
                audio_files = audio_files[: self.max_samples]

            self.samples = [
                {
                    "path": path,
                    "label": None,
                }
                for path in audio_files
            ]

            return

        # Original Speech Commands loading logic.
        unknown_files = []
        class_counts = {label: 0 for label in CORE_CLASSES}

        for class_dir in self.audio_dir.iterdir():
            if not class_dir.is_dir():
                continue

            if class_dir.name == "_background_noise_":
                continue

            class_name = class_dir.name

            for audio_path in class_dir.glob("*.wav"):
                relative_path = audio_path.relative_to(self.audio_dir)
                relative_path = relative_path.as_posix()

                if not self._belongs_to_split(relative_path):
                    continue

                if self.skip_anomalies and relative_path in self.anomalies:
                    continue

                if class_name in CORE_CLASSES:
                    if (
                        self.max_samples is None
                        or class_counts[class_name] < self.max_samples
                    ):
                        self.samples.append(
                            {
                                "path": audio_path,
                                "label": LABEL2ID[class_name],
                            }
                        )
                        class_counts[class_name] += 1
                else:
                    unknown_files.append(audio_path)

        self._add_unknown_and_silence(
            unknown_files,
            class_counts,
            add_silence=add_silence,
        )

    def _belongs_to_split(self, relative_path: str) -> bool:
        if self.split == "all":
            return True

        is_validation = relative_path in self.validation_files
        is_test = relative_path in self.test_files

        if self.split == "train":
            return not is_validation and not is_test

        if self.split == "val":
            return is_validation

        if self.split == "test":
            return is_test

        raise ValueError(f"Unknown split: {self.split}")

    def _add_unknown_and_silence(
        self,
        unknown_files: list[Path],
        class_counts: dict[str, int],
        add_silence: bool,
    ) -> None:
        random.shuffle(unknown_files)

        available_core_counts = [count for count in class_counts.values() if count > 0]

        if self.max_samples is not None:
            target_count = self.max_samples
        elif available_core_counts:
            target_count = min(available_core_counts)
        else:
            target_count = len(unknown_files)

        target_count = min(
            target_count,
            len(unknown_files),
        )

        # Unknown
        for path in unknown_files[:target_count]:
            self.samples.append(
                {
                    "path": path,
                    "label": LABEL2ID["unknown"],
                }
            )

        # Silence
        if add_silence:
            for _ in range(target_count):
                self.samples.append(
                    {
                        "path": None,
                        "label": LABEL2ID["silence"],
                    }
                )

    def _load_audio(self, path: Path | None) -> np.ndarray:
        if path is None:
            return self._sample_background_noise()

        audio, _ = sf.read(path)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return audio.astype(np.float32)

    def _sample_background_noise(self) -> np.ndarray:
        if not self.background_noises:
            return np.random.normal(0, 0.001, SAMPLE_LENGTH).astype(np.float32)

        noise = random.choice(self.background_noises)

        if len(noise) < SAMPLE_LENGTH:
            noise = np.pad(noise, (0, SAMPLE_LENGTH - len(noise)))

        start = random.randint(0, len(noise) - SAMPLE_LENGTH)

        audio = noise[start : start + SAMPLE_LENGTH]

        return (audio * random.uniform(0.0, 0.8)).astype(np.float32)

    @staticmethod
    def _fix_length(audio: np.ndarray) -> np.ndarray:
        if len(audio) < SAMPLE_LENGTH:
            audio = np.pad(
                audio,
                (0, SAMPLE_LENGTH - len(audio)),
                mode="constant",
            )

        return audio[:SAMPLE_LENGTH]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        audio = self._load_audio(sample["path"])
        audio = self._fix_length(audio)

        result = {
            "speech": torch.from_numpy(audio).unsqueeze(0),
        }

        if self.is_unlabeled:
            # Keep the original filename for Kaggle submission matching.
            result["path"] = sample["path"].name
        else:
            result["target"] = torch.tensor(sample["label"], dtype=torch.long)

        return result


class SpeechCommandsDatasetFactory:
    """
    Factory for creating Speech Commands or unlabeled Kaggle test datasets.

    Supported structures:

    1. Speech Commands:
        dataset/
            audio/
                yes/*.wav
                no/*.wav
                ...
            validation_list.txt
            testing_list.txt

    2. Kaggle test:
        dataset/
            audio/
                clip_001.wav
                clip_002.wav
                ...

    The format is detected automatically by default.

    Examples:
        factory = SpeechCommandsDatasetFactory("/path/to/dataset")

        train_dataset = factory.train()
        val_dataset = factory.val()
        test_dataset = factory.test()

        # Kaggle test returns speech and path, without target.
        for sample in test_dataset:
            audio = sample["speech"]
            filename = sample["path"]
    """

    VALID_SPLITS = {"train", "val", "test", "all"}

    def __init__(
        self,
        dataset_dir: str | Path,
        anomalies_file: str | Path | None = None,
        dataset_format: str = "auto",
    ):
        self.dataset_dir = Path(dataset_dir)

        self.audio_dir = self.dataset_dir / "audio"
        self.validation_list = self.dataset_dir / "validation_list.txt"
        self.test_list = self.dataset_dir / "testing_list.txt"

        if anomalies_file is None:
            anomalies_file = self.dataset_dir / "anomalies.json"

        self.anomalies_file = Path(anomalies_file)

        if dataset_format not in {
            "auto",
            "speech_commands",
            "kaggle_test",
        }:
            raise ValueError(
                f"Unknown dataset format: {dataset_format}. "
                "Expected 'auto', 'speech_commands' or 'kaggle_test'."
            )

        self._validate_paths()

        self.validation_files = self._load_split(self.validation_list)
        self.test_files = self._load_split(self.test_list)
        self.anomalies = self._load_anomalies(self.anomalies_file)

        # Detect format once so all factory methods use the same format.
        if dataset_format == "auto":
            self.dataset_format = self._detect_dataset_format()
        else:
            self.dataset_format = dataset_format

    def _validate_paths(self) -> None:
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        if not self.audio_dir.exists():
            raise FileNotFoundError(f"Audio directory not found: {self.audio_dir}")

    def _detect_dataset_format(self) -> str:
        direct_audio_files = list(self.audio_dir.glob("*.wav"))

        class_directories = [
            path
            for path in self.audio_dir.iterdir()
            if path.is_dir() and path.name != "_background_noise_"
        ]

        if direct_audio_files and not class_directories:
            return "kaggle_test"

        return "speech_commands"

    @staticmethod
    def _load_split(path: Path) -> set[str]:
        if not path.exists():
            return set()

        with path.open("r", encoding="utf-8") as file:
            return {line.strip() for line in file if line.strip()}

    @staticmethod
    def _load_anomalies(path: Path) -> set[str]:
        if not path.exists():
            return set()

        with path.open("r", encoding="utf-8") as file:
            anomalies = json.load(file)

        if not isinstance(anomalies, list):
            raise ValueError(f"Anomalies file must contain a JSON list: {path}")

        return {
            Path(item).as_posix()
            for item in anomalies
            if isinstance(item, str) and item.strip()
        }

    def create(
        self,
        split: str,
        max_samples: int | None = None,
        add_silence: bool = True,
        skip_anomalies: bool = False,
    ) -> SpeechCommandsDataset:
        if split not in self.VALID_SPLITS:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Expected one of {sorted(self.VALID_SPLITS)}."
            )

        return SpeechCommandsDataset(
            audio_dir=self.audio_dir,
            split=split,
            validation_files=self.validation_files,
            test_files=self.test_files,
            anomalies=self.anomalies,
            max_samples=max_samples,
            add_silence=add_silence,
            skip_anomalies=skip_anomalies,
            dataset_format=self.dataset_format,
        )

    def train(
        self,
        max_samples: int | None = None,
        add_silence: bool = True,
        skip_anomalies: bool = False,
    ) -> SpeechCommandsDataset:
        return self.create(
            split="train",
            max_samples=max_samples,
            add_silence=add_silence,
            skip_anomalies=skip_anomalies,
        )

    def val(
        self,
        max_samples: int | None = None,
        add_silence: bool = True,
        skip_anomalies: bool = False,
    ) -> SpeechCommandsDataset:
        return self.create(
            split="val",
            max_samples=max_samples,
            add_silence=add_silence,
            skip_anomalies=skip_anomalies,
        )

    def test(
        self,
        max_samples: int | None = None,
        add_silence: bool = True,
        skip_anomalies: bool = False,
    ) -> SpeechCommandsDataset:
        return self.create(
            split="test",
            max_samples=max_samples,
            add_silence=add_silence,
            skip_anomalies=skip_anomalies,
        )

    def all(
        self,
        max_samples: int | None = None,
        add_silence: bool = True,
        skip_anomalies: bool = False,
    ) -> SpeechCommandsDataset:
        return self.create(
            split="all",
            max_samples=max_samples,
            add_silence=add_silence,
            skip_anomalies=skip_anomalies,
        )
