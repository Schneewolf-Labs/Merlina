"""Small allowlist for hosted text training; local API config is not a security boundary."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import re

REPO = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?$')


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class SessionRequest(StrictModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=1024)


class ProjectRequest(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    goal: str = Field(default='', max_length=10000)


class ProjectUpdate(StrictModel):
    name: str = Field(default=None, min_length=1, max_length=120)
    goal: str = Field(default=None, max_length=10000)


class DatasetUpload(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=10 * 1024 * 1024)


class HostedSource(StrictModel):
    source_type: Literal['huggingface', 'upload']
    repo_id: str | None = None
    dataset_id: str | None = None
    split: Literal['train', 'test', 'validation'] = 'train'

    @model_validator(mode='after')
    def source_matches(self):
        if self.source_type == 'huggingface':
            if not self.repo_id or not REPO.fullmatch(self.repo_id) or '..' in self.repo_id or self.dataset_id is not None:
                raise ValueError('A Hugging Face repository ID is required; paths and URLs are forbidden')
        elif not self.dataset_id or not re.fullmatch('[a-f0-9]{32}', self.dataset_id) or self.repo_id is not None:
            raise ValueError('An owned dataset ID is required')
        return self


class HostedDatasetFormat(StrictModel):
    format_type: Literal['chatml', 'llama3', 'mistral', 'qwen3', 'tokenizer', 'raw'] = 'chatml'


class HostedDataset(StrictModel):
    source: HostedSource
    format: HostedDatasetFormat = Field(default_factory=HostedDatasetFormat)
    test_size: float = Field(default=.05, ge=.001, le=.5)


class HostedJobConfig(StrictModel):
    base_model: str = Field(min_length=1, max_length=200)
    dataset: HostedDataset
    training_mode: Literal['sft', 'orpo', 'dpo', 'simpo', 'cpo', 'ipo', 'kto'] = 'sft'
    model_type: Literal['causal_lm'] = 'causal_lm'
    num_epochs: int = Field(default=1, ge=1, le=10)
    learning_rate: float = Field(default=5e-6, ge=1e-8, le=1e-3)
    batch_size: int = Field(default=1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=16, ge=1, le=128)
    max_length: int = Field(default=2048, ge=512, le=8192)
    # None = auto: each row's prompt gets max_length minus its response (response capped
    # at half); over-long prompts keep their end. See src/prompt_budget.py.
    max_prompt_length: int | None = Field(default=None, ge=256, le=8192)
    lora_r: int = Field(default=16, ge=8, le=128)
    lora_alpha: int = Field(default=32, ge=8, le=256)

    @field_validator('base_model')
    @classmethod
    def hub_model_only(cls, value):
        if not REPO.fullmatch(value) or '..' in value:
            raise ValueError('Use a Hub model ID; paths and URLs are forbidden')
        return value

    @model_validator(mode='after')
    def lengths(self):
        if self.max_prompt_length is not None and self.max_prompt_length >= self.max_length:
            raise ValueError('max_prompt_length must be less than max_length')
        return self


class JobRequest(StrictModel):
    config: HostedJobConfig
