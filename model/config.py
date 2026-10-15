from dataclasses import asdict, dataclass


@dataclass
class JBRConfig:
    vocab_size: int
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    chunk_size: int = 512

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "JBRConfig":
        return cls(**raw)
