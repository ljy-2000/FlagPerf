from transformers import AutoModelForCausalLM


def create_model(config):
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    model = AutoModelForCausalLM.from_pretrained(model_name).eval().cuda().float()
    if config.fp16:
        model.half()

    return model
