from dataclasses import dataclass

@dataclass
class OXEDatasetConfig:
    bucket: str = "gs://x-embodiment-imporvement"
    root_prefix: str = "oxe_v1_0"

    @property
    def base_path(self) -> str:
        return f"{self.bucket}/{self.root_prefix}"
