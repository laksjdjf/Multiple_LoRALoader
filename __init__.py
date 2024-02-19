from .node import create_class
import os
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(CURRENT_DIR, "config.txt"), "r") as f:
    config = f.read()
num_loras = [int(i) for i in config.replace(" ", "").split(",")]

NODE_CLASS_MAPPINGS = {
    f"MultipleLoraLoader{i}": create_class(i) for i in num_loras
}

NODE_CLASS_MAPPINGS.update({
    f"MultipleLoraLoaderModelOnly{i}": create_class(i, model_only=True) for i in num_loras
})

__all__ = [NODE_CLASS_MAPPINGS]