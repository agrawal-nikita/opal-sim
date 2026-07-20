# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
import os
from pathlib import Path
import re
from typing import Dict, Any, Optional
from pprint import pprint, pformat

from opal.core.datatypes import STR_DTYPE_TO_BYTES


class OpalModelConfig:
    """LLM model class
    Args:
        hf_url (url, optional): Initialize all model parameters from Hugging Face model config file
        config_file (str, optional): Initialize all model parameters from a local model config file
        config_dict (dict, optional): Initialize all model parameters from a dictionary
    """

    @classmethod
    def init_from_huggingface(cls, hf_url: str) -> Optional[OpalModelConfig]:
        return OpalModelConfig(hf_url=hf_url, config_dir=None)

    @classmethod
    def init_from_config_dir(cls, config_file: str) -> Optional[OpalModelConfig]:
        return OpalModelConfig(hf_url=None, config_dir=config_file)

    def __init__(self, hf_url: Optional[str], config_dir: Optional[str]):
        self.logger = logging.getLogger("OpalModelConfig")
        self.logger.debug(f"init params = {hf_url}, {config_dir}")
        if hf_url is not None:
            from transformers import AutoConfig, PretrainedConfig

            self.hg_config: PretrainedConfig = AutoConfig.from_pretrained(hf_url)
            self.config_dict = self.hg_config.to_dict()
            self.config_dict["model_name"] = self._get_model_name_from_url(hf_url)
            self.hf_name = hf_url
            from transformers.utils import cached_file

            # over-ride this params here so that we can calculate num of params
            self._config_file: Path = Path(cached_file(hf_url, "config.json")).resolve()
            self._config_dir: Path = self._config_file.parent
        elif config_dir is not None:
            self._config_file = Path(os.path.join(config_dir, "config.json"))
            self._config_dir = self._config_file.parent
            import json

            self.config_dict = json.load(open(self._config_file))
            if "model_name" not in self.config_dict:
                self.config_dict["model_name"] = self._get_model_name_from_url(str(config_dir))

        else:
            raise ValueError("Must provide hf_url, or the config dir")

        self.model_name = self.config_dict["model_name"]
        assert self.config_dict is not None
        print(self.config_dict)
        self.vocab_size = int(self.config_dict["vocab_size"])
        self.hidden_size = int(self.config_dict["hidden_size"])
        self._set_num_attention_head(self.config_dict)
        self.set_num_hidden_layers(self.config_dict)
        self.kv_head_size = self._get_kv_head_dim(self.config_dict)
        self.num_key_value_heads = int(self.config_dict["num_key_value_heads"])
        self.max_position_embeddings = int(self.config_dict.get("max_position_embeddings"))
        # transformers >= 4.55 renamed "torch_dtype" to "dtype" in .to_dict() output
        dtype_name = self.config_dict.get("torch_dtype") or self.config_dict.get("dtype")
        if dtype_name in STR_DTYPE_TO_BYTES:
            self.torch_dtype_bytes = STR_DTYPE_TO_BYTES[dtype_name]
            self.torch_dtype_name = dtype_name
        else:
            raise ValueError(f"Unknown dtype '{dtype_name}'. Supported types: {list(STR_DTYPE_TO_BYTES.keys())}")
        # derivative values
        self.key_bytes = self.kv_head_size * self.num_key_value_heads * self.num_hidden_layers * self.torch_dtype_bytes
        self.key_value_bytes = 2 * self.key_bytes
        self.logger.debug(
            "key_value bytes for model {} is {} bytes ({:.2f} KiB) ".format(
                self.model_name, self.key_value_bytes, self.key_value_bytes >> 10
            )
        )
        self.logger.debug(
            "max number of tokens are {} and memory needed {:.2f} MiB".format(
                self.max_position_embeddings, (self.key_value_bytes * self.max_position_embeddings) >> 20
            )
        )

        self.model_params = self._estimate_params_from_config_universal()

    def get_kvc_bytes(self, tokens: int) -> int:
        """Returns the size of the key value in bytes for "tokens".
        Why is this a function? Because for hybrid models it will include
        the SSM size also.

        Args:
            tokens (int): Number of tokens

        Returns:
            int: size in bytes for kv cache data for "tokens" tokens.
        """
        return tokens * self.key_value_bytes

    def get_kvc_tokens(self, size_bytes: int) -> int:
        """Returns the numbers of tokens that can be saved in the passed number of bytes
        Why is this a function? Because for hybrid models it will include
        the SSM size also.

        Args:
            size_bytes (int): Number of bytes

        Returns:
            int: number of tokens that can be saved in it ß.
        """
        return size_bytes // self.key_value_bytes

    def _set_num_attention_head(self, config: dir):
        keys = config.keys()
        if "num_attention_heads" in keys:
            self.num_attention_heads = int(self.config_dict["num_attention_heads"])
        elif "n_head" in keys:
            self.num_attention_heads = int(self.config_dict["n_head"])
        else:
            raise Exception("I dont know how to get num_attention_heads for the model")

    def set_num_hidden_layers(self, config: dir):
        keys = config.keys()
        if "num_hidden_layers" in keys:
            self.num_hidden_layers = int(self.config_dict["num_hidden_layers"])
        elif "n_layer" in keys:
            self.num_hidden_layers = int(self.config_dict["n_layer"])
        else:
            raise Exception("I dont know how to get num_attention_heads for the model")

    def __str__(self):
        return pformat(self.config_dict)

    def toJSON(self):
        import json

        return json.dumps(self.config_dict, indent=4)

    # code copied ana adapted from from config.py file from vLLM
    def _get_kv_head_dim(self, model_config) -> int:
        # if the hidden embedding dimmension is given then use it
        if "head_dim" in model_config:
            return model_config["head_dim"]
        # otherwise it will be hidden_size (or model_size) / num_attention_heads
        return model_config["hidden_size"] // self.num_attention_heads

    def _get_model_name_from_url(self, hf_url):
        parts = hf_url.split("/")
        # return the last one
        return parts[-1]

        # for p in parts:
        #     if p in ["https:", "", "huggingface.co", "meta-llama", "blob", "main", "config.json"]:
        #         parts.remove(p)
        # name = parts[-1].replace(".json", "")
        # return name

    def _estimate_params_from_model(self, config_folder):
        """
        Estimate total parameters from a local Hugging Face config folder.
        Does not require weights, works offline. BUT allocates DRAM to hold
        the "randomly" generated model.

        # Example usage:
           config_folder = "/model_configs/llama-3.3-70B-instruct"  # folder containing config.json
           estimate_params_from_local_config(config_folder)
        """
        from transformers import AutoConfig, AutoModel
        import os

        if not os.path.isdir(config_folder):
            raise ValueError(f"{config_folder} is not a valid directory.")

        # Load the config only
        config = AutoConfig.from_pretrained(config_folder)

        # Create a model with random weights based on config
        model = AutoModel.from_config(config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Model config: {config_folder}")
        print(f"Total parameters: {total_params/1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params/1e6:.2f}M")

        return total_params, trainable_params

    def _estimate_params_from_config(self):
        """
        Universal Hugging Face transformer parameter estimator.
        Works offline, does not instantiate model, suitable for huge models.

        # Example usage
            config_file = "/model_configs/llama-3.3-70B-instruct/config.json"
            estimate_transformer_params(config_file)
        """

        def estimate_ffn_intermediate_size(config):
            """
            From Gemini: Generic way to calculate the FFN size for a give
            model config from hugging face.
            """
            """
            Changes below: before it was 
                if hasattr(config, "intermediate_size") and config.intermediate_size:
                -> which checks if a class "attribute" is there or not 
            while I have switched to using generic dict for the config, hence, need to 
            check for if the key exist in the dict 
            """

            # 1. Check if it's already explicitly defined
            if "intermediate_size" in config:
                return config["intermediate_size"]

            # 2. Generic Llama-style calculation logic
            hidden_size = config["hidden_size"]

            # Standard SwiGLU adjustment (8/3 factor)
            # Llama 3 70B uses 3.5, others use ~2.66
            factor = 8 / 3

            # Check for specific multipliers in config
            if "ffn_dim_multiplier" in config:
                factor *= config["ffn_dim_multiplier"]

            intermediate_size = int(factor * hidden_size)

            # 3. Rounding to multiple (alignment)
            multiple_of = config.get("multiple_of", 256)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

            return intermediate_size

        config = self.config_dict
        # Extract hyperparameters with fallbacks
        hidden_size = config.get("hidden_size") or config.get("d_model")
        num_layers = config.get("num_hidden_layers") or config.get("n_layer")
        intermediate_size = estimate_ffn_intermediate_size(config)
        vocab_size = config.get("vocab_size") or config.get("vocab_size")
        num_attention_heads = config.get("num_attention_heads") or config.get("n_head")
        head_dims = self._get_kv_head_dim(self.config_dict)
        num_kv_heads = config.get("num_key_value_heads")

        if None in [hidden_size, num_layers, num_attention_heads]:
            raise ValueError("Config missing essential transformer hyperparameters.")

        # Set a default intermediate size if missing (common for GPT variants)
        if intermediate_size is None:
            intermediate_size = hidden_size * 4

        # 1️⃣ Embedding layer
        embedding_params = vocab_size * hidden_size
        print(f" embedding layer is {embedding_params}")

        # 2️⃣ (A) Attention per layer: Q, K, V, output projection
        Q_size = hidden_size * num_attention_heads * head_dims
        KV_size = 2 * hidden_size * num_kv_heads * head_dims
        output_projection = hidden_size * hidden_size
        attention_params_per_layer = Q_size + KV_size + output_projection

        # 3️⃣ (B) Feed-forward layer per transformer block
        ff_params_per_layer = 3 * hidden_size * intermediate_size

        # 4️⃣ Total transformer parameters = 2A + 2B
        transformer_params = num_layers * (attention_params_per_layer + ff_params_per_layer)

        # 5️⃣ Output layer (language modeling head)
        tie_embeddings = config.get("tie_word_embeddings", True)  # Default is often True in small models
        if tie_embeddings:
            output_params = 0
            print("Embeddings are tied; skipping output layer count.")
        else:
            output_params = vocab_size * hidden_size

        # Total parameters
        total_params = embedding_params + transformer_params + output_params

        print(f"Estimated total parameters: {total_params/1e6:.2f}M")
        print(f"Embedding layer: {embedding_params/1e6:.2f}M")
        print(f"Transformer layers: {transformer_params/1e6:.2f}M")
        print(f"Output layer: {output_params/1e6:.2f}M")

        return total_params

    def _estimate_params_from_config_universal(self):
        import math

        config = self.config_dict

        # 1. ARCHITECTURE DETECTION
        model_type = config.get("model_type", "").lower()
        num_experts = config.get("num_experts", 0)
        num_dense_layers = config.get("num_dense_layers", 0)

        # Precise Mamba detection
        is_mamba_family = "mamba" in model_type or "jamba" in model_type
        is_moe = num_experts > 0

        # 2. CORE HYPERPARAMETERS
        hidden_size = config.get("hidden_size") or config.get("d_model")
        num_layers = config.get("num_hidden_layers") or config.get("n_layer")
        vocab_size = config.get("vocab_size")
        tie_embeddings = config.get("tie_word_embeddings", False)

        # 3. EMBEDDING & LM HEAD
        embedding_params = vocab_size * hidden_size
        output_params = 0 if tie_embeddings else (vocab_size * hidden_size)

        total_transformer_params = 0
        active_transformer_params = 0

        # 4. LAYER-BY-LAYER CALCULATION
        for i in range(num_layers):
            # A) ATTENTION OR MAMBA "COMMUNICATION" LAYER
            is_layer_mamba = False
            if "jamba" in model_type:
                # Jamba interleaves: 1 Attention layer for every N Mamba layers
                attn_period = config.get("attn_layer_period", 8)
                is_layer_mamba = i % attn_period != 0
            elif "mamba" in model_type:
                is_layer_mamba = True

            if is_layer_mamba:
                # --- MAMBA SSM MATH ---
                d_state = config.get("state_size", 16)
                d_conv = config.get("conv_kernel", 4)
                expand = config.get("expand", 2)
                d_inner = int(expand * hidden_size)

                in_proj = hidden_size * (d_inner * 2)  # Proj into SSM + Gating branch
                out_proj = d_inner * hidden_size
                conv_params = d_inner * d_conv

                # SSM (A, B, C, delta)
                dt_rank = (
                    math.ceil(hidden_size / 16)
                    if config.get("time_step_rank") == "auto"
                    else config.get("time_step_rank", 1)
                )
                ssm_params = (d_inner * d_state) + (d_inner * dt_rank) + (dt_rank * hidden_size)

                comm_params = in_proj + out_proj + conv_params + ssm_params
            else:
                # --- TRANSFORMER ATTENTION MATH ---
                num_heads = config.get("num_attention_heads", 1)
                num_kv = config.get("num_key_value_heads", num_heads)
                head_dim = config.get("head_dim", hidden_size // num_heads)

                q_proj = hidden_size * (num_heads * head_dim)
                kv_proj = 2 * (num_kv * head_dim) * hidden_size
                o_proj = hidden_size * hidden_size
                comm_params = q_proj + kv_proj + o_proj

            # B) MLP / EXPERT "REASONING" LAYER
            current_layer_is_moe = is_moe and (i >= num_dense_layers)

            # Expert logic for Jamba-style models
            if "jamba" in model_type:
                # Jamba often has experts every N layers (expert_layer_period)
                exp_period = config.get("expert_layer_period", 2)
                current_layer_is_moe = i % exp_period == 0

            if current_layer_is_moe:
                expert_size = config.get("moe_intermediate_size") or config.get("intermediate_size")
                k = config.get("num_experts_per_tok", 1)
                router = hidden_size * num_experts

                layer_total_mlp = num_experts * (3 * hidden_size * expert_size) + router
                layer_active_mlp = k * (3 * hidden_size * expert_size) + router
            else:
                # Dense SwiGLU MLP
                intermediate_size = config.get("intermediate_size", hidden_size * 4)
                mlp_params = 3 * hidden_size * intermediate_size
                layer_total_mlp = mlp_params
                layer_active_mlp = mlp_params

            # C) NORM & SUMMARIZE
            norms = 2 * hidden_size
            total_transformer_params += comm_params + layer_total_mlp + norms
            active_transformer_params += comm_params + layer_active_mlp + norms

        # 5. FINAL RESULTS
        total = embedding_params + total_transformer_params + output_params
        active = embedding_params + active_transformer_params + output_params

        return total

    def _calculate_model_params_cdir(self):
        """
        There are a couple of ways to deal with it

        1. model = AutoModel.from_pretrained()   - will load the model and config
        2. config = AutoConfig.from_pretrained(config_path)   - just load the config
          but this does not have the parameter estimation what we need
          2026-02-27 11:56:12 - why?

        3. model = AutoModel.from_config(model_config_path) - randomly assigns weights
        """
        pass

    def get_model_params(self):
        return self.model_params

    def get_model_name(self):
        return self.model_name


def get_config_folder_for_model(config_dir: str, model_name: str):
    from pathlib import Path

    root_path = Path(config_dir)
    print(f"Scanning model/model_params/config_dir {config_dir} for the model {model_name}")
    # rglob("*") recursively finds all files and folders
    for path in root_path.rglob("*"):
        if path.is_dir():
            print(f"\t...processing location: {path.resolve()}")
            if model_name in path.parts:
                # we found a match, return
                return path
    print(f"WARNING: no suitable local folder found in {config_dir} for the model {model_name}")
    return None


def get_model(
    *,
    name: str | None = None,
    config_dir: str | None = None,
    hf_url: str | None = None,
):
    if config_dir is not None and hf_url is not None:
        raise ValueError(
            f"Provide exactly one config source either hf_url {hf_url} or config_dir {config_dir}, not both."
        )

    if hf_url is not None:
        return OpalModelConfig.init_from_huggingface(hf_url)
    else:
        return OpalModelConfig.init_from_config_dir(get_config_folder_for_model(config_dir, model_name=name))
