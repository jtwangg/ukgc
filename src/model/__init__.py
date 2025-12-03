from src.model.llm import LLM
from src.model.llm_ds import LLMDS
from src.model.pt_llm import PromptTuningLLM
from src.model.pt_llm_ds import PromptTuningLLMDS
from src.model.graph_llm import GraphLLM
from src.model.graph_llm_ds import GraphLLMDS
from src.model.llm_newloss import LLM_newloss
from src.model.graph_llm_ds_customtrainer import GraphLLMDSCT


load_model = {
    "llm": LLM,
    "llm_ds": LLMDS,
    "inference_llm": LLM,
    "pt_llm": PromptTuningLLM,
    "pt_llm_ds": PromptTuningLLMDS,
    "graph_llm": GraphLLM,
    "graph_llm_ds": GraphLLMDS,
    "llm_newloss": LLM_newloss,
    'graph_llm_ds_customtrainer': GraphLLMDSCT,
}

# Replace the following with the model paths
llama_model_path = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "7b_chat": "/seu_share/home/qiguilin/220236147/huggingface_models/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
    "8b": "/seu_share/home/qiguilin/220236147/huggingface_models/Llama-3.1-8B-Instruct",
    "3b": "/seu_share/home/qiguilin/220236147/huggingface_models/Llama-3.2-3B-Instruct",
}
