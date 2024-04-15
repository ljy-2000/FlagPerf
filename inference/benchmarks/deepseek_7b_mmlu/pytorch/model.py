from transformers import AutoModelForCausalLM


def create_model(config):
    model = AutoModelForCausalLM.from_pretrained(model_name).eval().cuda().float()
    if config.fp16:
        model.half()

    return model
