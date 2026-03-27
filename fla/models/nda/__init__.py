
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.nda.configuration_nda import NDAConfig
from fla.models.nda.modeling_nda import NDAForCausalLM, NDAModel

AutoConfig.register(NDAConfig.model_type, NDAConfig, exist_ok=True)
AutoModel.register(NDAConfig, NDAModel, exist_ok=True)
AutoModelForCausalLM.register(NDAConfig, NDAForCausalLM, exist_ok=True)

__all__ = ['NDAConfig', 'NDAForCausalLM', 'NDAModel']
