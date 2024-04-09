from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class ModelArguments_finetune:
    model_name_or_path: Optional[str] = field(default="LLaVA-Pretrain/checkpoints/vicuna-7b-v1.5")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)  # Assuming tuning is enabled since --tune_mm_mlp_adapter True in the ds_config
    vision_tower: Optional[str] = field(default="LLaVA-Pretrain/openai/clip-vit-large-patch14-336")
    mm_vision_select_layer: Optional[int] = field(default=-2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default="/home/LLaVA/LLaVA-main/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin")
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_patch_merge_type: Optional[str] = field(default='flat')  # Assuming this remains unchanged
    mm_vision_select_feature: Optional[str] = field(default="patch")  # Assuming this remains unchanged



@dataclass
class DataArguments_finetune:
    data_path: str = field(default="/root/llava1.5_finetune/llava_v1_5_mix665k.json",
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = True  # Assuming lazy_preprocess is enabled since --lazy_preprocess True in the ds_config
    is_multimodal: bool = False  # Assuming this remains unchanged
    image_folder: Optional[str] = field(default="/root/llava1.5_finetune/data")
    image_aspect_ratio: str = 'pad'


@dataclass
class TrainingArguments_finetune(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    output_dir: str = field(default="./checkpoints/llava-v1.5-7b")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=16)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    evaluation_strategy: str = field(default="no")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=50000)
    save_total_limit: int = field(default=1)
    learning_rate: float = field(default=2e-5)
    weight_decay: float = field(default=0.)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=1)
    bf16: bool = field(default=True)
    tf32: bool = field(default=True)
    model_max_length: int = field(default=2048)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    report_to: str = field(default="none")
    deepspeed: str = field(default="config/ds_config_finetune.json")
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"