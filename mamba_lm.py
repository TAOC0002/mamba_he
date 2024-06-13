from dataclasses import dataclass, fields, asdict
import torch.nn as nn
from mamba import Mamba, MambaConfig, RMSNorm

@dataclass
class MambaLMConfig(MambaConfig):
    vocab_size: int = 32000
    pad_vocab_size_multiple: int = 8

    def __post_init__(self):
        super().__post_init__()

        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

    def to_mamba_config(self) -> MambaConfig:
        mamba_config_fields = {field.name for field in fields(MambaConfig)}
        filtered_dict = {k: v for k, v in asdict(self).items() if k in mamba_config_fields}
        return MambaConfig(**filtered_dict)

class MambaLM(nn.Module):
    def __init__(self, lm_config: MambaLMConfig):
        super().__init__()
        self.lm_config = lm_config
        self.config = lm_config.to_mamba_config()

        self.embedding = nn.Embedding(self.lm_config.vocab_size, self.config.d_model)
        self.mamba = Mamba(self.config)
        self.norm_f = RMSNorm(self.config.d_model)

        self.lm_head = nn.Linear(self.config.d_model, self.lm_config.vocab_size, bias=False)
        # self.lm_head.weight = self.embedding.weight # do not initialize as such -> this will reduce performance

    def forward(self, tokens):
        # tokens : (B, L)

        # logits : (B, L, vocab_size)

        x = self.embedding(tokens)

        x = self.mamba(x)
        x = self.norm_f(x)

        logits = self.lm_head(x)

        return logits 