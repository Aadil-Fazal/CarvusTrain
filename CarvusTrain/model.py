"""Model Builder — Enhanced model creation with architecture selection and auto-configuration."""
import os
from typing import Any, Dict, List, Optional, Union

from .configuration import CarvusConfig, ModelConfig, TrainingConfig
from .constants import DEFAULT_MODEL_EXTENSION
from .dataset import Dataset, TextDataset
from .evaluation import Benchmarker, Evaluator
from .exceptions import ModelError
from .exporter import ModelExporter
from .inference import InferenceEngine
from .logger import logger
from .memory import KnowledgeBase, LearningValidator
from .parser import CarvusTrainParser, DataParser
from .tokenizer import Tokenizer
from .trainer import Trainer
from .utils import get_system_info

SUPPORTED_ARCHITECTURES = ["transformer", "neural_network", "cnn", "rnn", "lstm", "custom"]
SUPPORTED_MODES = ["general", "programmer", "researcher", "teacher", "creative", "security", "data_scientist"]


class Model:
    """Primary high-level AI Model interface in CarvusTrain.
    
    Features:
    - Multiple architectures: transformer, neural_network, cnn, rnn, lstm, custom
    - AI Coding Modes: general, programmer, researcher, teacher, creative, security, data_scientist
    - Auto AI Trainer with smart defaults
    - Learning validation and knowledge comprehension
    - RAG-ready retrieval augmented generation
    - Plugin system for extensions
    """

    def __init__(
        self,
        name: str = "Carvus",
        architecture: str = "transformer",
        parameters: str = "base",
        mode: str = "general",
        config: Optional[Union[CarvusConfig, ModelConfig, Dict[str, Any]]] = None,
        auto_load: bool = True,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.architecture = architecture.lower() if architecture in SUPPORTED_ARCHITECTURES else "transformer"
        self.parameters = parameters
        self._mode = mode.lower() if mode.lower() in SUPPORTED_MODES else "general"

        if isinstance(config, CarvusConfig):
            self.config = config
        elif isinstance(config, ModelConfig):
            self.config = CarvusConfig(model=config)
        elif isinstance(config, dict):
            self.config = CarvusConfig.from_dict(config)
        else:
            self.config = CarvusConfig(model=ModelConfig(name=name, **kwargs))

        self.tokenizer = Tokenizer.create("word")
        self.knowledge_base = KnowledgeBase()
        self.learning_validator = LearningValidator(self.knowledge_base)
        self.inference_engine = InferenceEngine(self.knowledge_base, self.config.inference)
        self.trainer: Optional[Trainer] = None
        self._last_answer: str = ""
        self._is_training: bool = False
        self._memory_enabled: bool = False
        self._conversation_history: List[Dict[str, str]] = []
        self._plugins: Dict[str, Any] = {}
        self._personality: Dict[str, Any] = {}

        if auto_load and os.path.exists("train.ct"):
            try:
                parsed = CarvusTrainParser.parse_file("train.ct")
                if parsed.model_config.get("Name"):
                    self.name = parsed.model_config.get("Name")
                if parsed.knowledge:
                    self.knowledge_base.add_facts(parsed.knowledge)
                    logger.info(f"Loaded {len(parsed.knowledge)} knowledge facts from train.ct")
                if parsed.training_config.get("LearnCheck", "").lower() == "true":
                    logger.info("LearnCheck enabled - learning validation active")
            except Exception as e:
                logger.debug(f"Could not load train.ct: {e}")

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, new_mode: str) -> None:
        new_mode = new_mode.lower()
        if new_mode in SUPPORTED_MODES:
            self._mode = new_mode
            logger.info(f"Model mode set to '{new_mode}'")

    @property
    def answer(self) -> str:
        return self._last_answer

    def set_training_mode(self, training: bool = True) -> None:
        self._is_training = training

    def enable_memory(self, enabled: bool = True) -> None:
        self._memory_enabled = enabled
        logger.info(f"Long-term memory {'enabled' if enabled else 'disabled'}")

    def set_personality(self, style: str = "professional", humor: int = 20, creativity: int = 80) -> None:
        self._personality = {"style": style, "humor": max(0, min(100, humor)), "creativity": max(0, min(100, creativity))}
        logger.info(f"Personality set: style={style}, humor={self._personality['humor']}, creativity={self._personality['creativity']}")

    def install_plugin(self, name: str, plugin: Any) -> None:
        self._plugins[name] = plugin
        logger.info(f"Plugin '{name}' installed")

    def auto_train(self, data: Optional[Union[str, Dataset, List[str]]] = None, auto_optimize: bool = True) -> Dict[str, Any]:
        """Auto AI Trainer — automatically determines best training strategy."""
        logger.info("Auto AI Trainer: Analyzing data and configuring optimal training strategy...")
        if auto_optimize:
            if isinstance(data, str) and os.path.isfile(data):
                size = os.path.getsize(data)
                batch_size = 8 if size > 10_000_000 else 16 if size > 1_000_000 else 32
                lr = 1e-4 if size > 10_000_000 else 5e-4 if size > 1_000_000 else 1e-3
            elif isinstance(data, list):
                batch_size = 8 if len(data) > 1000 else 32
                lr = 1e-4 if len(data) > 1000 else 1e-3
            else:
                batch_size = 32
                lr = 1e-3
            logger.info(f"Auto-optimized: batch_size={batch_size}, lr={lr}")
            return self.train(data=data, batch_size=batch_size, learning_rate=lr)
        return self.train(data=data)

    def train(
        self,
        data: Optional[Union[str, Dataset, List[str]]] = None,
        method: str = "normal",
        duration: Union[int, str] = 10,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        optimizer: Optional[str] = None,
        device: str = "auto",
        cpu: bool = False,
        cuda: bool = False,
        gpu: bool = False,
        mixed_precision: bool = False,
        workers: int = 0,
        shuffle: bool = True,
        seed: int = 42,
        checkpoint: bool = True,
        resume: bool = False,
        validation_split: float = 0.1,
        early_stopping: Optional[int] = None,
        dropout: Optional[float] = None,
        weight_decay: Optional[float] = None,
        gradient_clipping: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        dev = "cpu" if cpu else ("cuda" if (cuda or gpu) else device)
        train_cfg = self.config.training
        train_cfg.method = method
        train_cfg.duration = duration
        if epochs is not None:
            train_cfg.epochs = epochs
        if batch_size is not None:
            train_cfg.batch_size = batch_size
        if learning_rate is not None:
            train_cfg.learning_rate = learning_rate
        if optimizer is not None:
            train_cfg.optimizer = optimizer
        train_cfg.device = dev
        train_cfg.mixed_precision = mixed_precision
        train_cfg.workers = workers
        train_cfg.shuffle = shuffle
        train_cfg.seed = seed
        train_cfg.checkpoint = checkpoint
        train_cfg.resume = resume
        train_cfg.validation_split = validation_split
        train_cfg.early_stopping = early_stopping
        if weight_decay is not None:
            train_cfg.weight_decay = weight_decay
        if gradient_clipping is not None:
            train_cfg.gradient_clipping = gradient_clipping

        dataset_obj = self._prepare_dataset(data)
        self.trainer = Trainer(model=self, config=train_cfg)
        train_ds, val_ds = dataset_obj.split(val_ratio=validation_split, seed=seed) if validation_split > 0 else (dataset_obj, None)

        history = self.trainer.train(dataset=train_ds, val_dataset=val_ds, epochs=epochs if epochs is not None else duration, batch_size=batch_size, method=method)

        if self.learning_validator:
            vs = self.learning_validator.get_validation_summary()
            logger.info(f"Learning validation: {vs['status']} (accuracy: {vs.get('average_accuracy', 0):.4f})")
        return history

    def _prepare_dataset(self, data: Optional[Union[str, Dataset, List[str]]]) -> Dataset:
        if data is None:
            if os.path.exists("train.ct"):
                data = "train.ct"
            else:
                return TextDataset.from_texts(["Carvus is an advanced AI assistant created with CarvusTrain."])
        if isinstance(data, Dataset):
            texts = data.get_all_texts() if hasattr(data, "get_all_texts") else [str(s.get("text", "")) for s in data]
            self.knowledge_base.add_facts(texts)
            return data
        if isinstance(data, list):
            self.knowledge_base.add_facts(data)
            return TextDataset.from_texts(data)
        if isinstance(data, str):
            if os.path.exists(data):
                ext = os.path.splitext(data)[1].lower()
                if ext in (".ct", ".cl") or data.endswith(".ct") or data.endswith(".cl"):
                    parsed = CarvusTrainParser.parse_file(data)
                    texts = list(parsed.knowledge)
                    self.knowledge_base.add_facts(parsed.knowledge)
                    # Learn from [Examples] section — these are teaching examples
                    if parsed.examples:
                        for ex in parsed.examples:
                            self.learn(ex)
                            texts.append(ex)
                        logger.info(f"Learned {len(parsed.examples)} examples from '{data}'")
                    return TextDataset.from_texts(texts if texts else [data])
                else:
                    parsed_records = DataParser.parse(data)
                    texts = [str(r.get("text", "")) for r in parsed_records]
                    self.knowledge_base.add_facts(texts)
                    return Dataset(parsed_records)
            else:
                self.knowledge_base.add_fact(data)
                return TextDataset.from_texts([data])
        return TextDataset.from_texts([str(data)])

    def ask(self, question: str) -> str:
        ans = self.inference_engine.ask(question)
        self._last_answer = ans
        if self._memory_enabled:
            self._conversation_history.append({"role": "user", "content": question})
            self._conversation_history.append({"role": "assistant", "content": ans})
        return ans

    def predict(self, input_text: str) -> str:
        return self.ask(input_text)

    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs: Any) -> str:
        return self.inference_engine.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def chat(self, message: str) -> str:
        prefix = ""
        if self._mode == "programmer":
            prefix = "[Programmer Mode] "
        elif self._mode == "teacher":
            prefix = "[Teacher Mode] "
        elif self._mode == "researcher":
            prefix = "[Research Mode] "
        elif self._mode == "creative":
            prefix = "[Creative Mode] "
        elif self._mode == "security":
            prefix = "[Security Mode] "
        elif self._mode == "data_scientist":
            prefix = "[Data Science Mode] "
        ans = self.inference_engine.chat(prefix + message)
        self._last_answer = ans
        return ans

    def learn(self, text: Union[str, List[str]]) -> None:
        if isinstance(text, str):
            texts = [text]
        elif isinstance(text, list):
            texts = text
        else:
            texts = [str(text)]
        self.knowledge_base.add_facts(texts)
        self.tokenizer.train_on_texts(texts)
        if len(texts) >= 2:
            generated = []
            for t in texts[:3]:
                generated.append(self.inference_engine.ask(t[:100]))
            val_metrics = self.learning_validator.validate_learning(texts[:3], generated)
            logger.info(f"Learning validation: accuracy={val_metrics['accuracy']:.4f}, comprehension={val_metrics['comprehension_score']:.4f}")
        logger.info(f"Learned new knowledge. Total facts: {len(self.knowledge_base)}")

    def learn_patterns(self, texts: List[str]) -> float:
        self.tokenizer.train_on_texts(texts)
        self.knowledge_base.add_facts(texts)
        code_facts_extracted = 0
        for text in texts:
            if self._text_contains_code(text):
                lang = self.knowledge_base._detect_language(text)
                if lang:
                    patterns = self.knowledge_base._extract_code_patterns(text, lang)
                    if patterns:
                        code_facts_extracted += len(patterns)
        kb_size = len(self.knowledge_base)
        if kb_size == 0:
            return 0.5
        knowledge_coverage = min(1.0, kb_size / 200.0)
        base_loss = max(0.05, 0.5 * (1.0 - knowledge_coverage))
        if code_facts_extracted > 0:
            code_bonus = min(0.2, code_facts_extracted * 0.02)
            base_loss = max(0.05, base_loss - code_bonus)
        if self.learning_validator:
            avg_acc = self.knowledge_base.get_average_accuracy()
            if avg_acc > 0:
                base_loss *= max(0.3, 1.0 - avg_acc)
        return float(base_loss)

    @staticmethod
    def _text_contains_code(text: str) -> bool:
        import re
        indicators = [r"def\s+\w+\s*\(", r"class\s+\w+", r"function\s+\w+\s*\(", r"fn\s+\w+", r"func\s+\w+", r"#include", r"=>", r"->", r"public\s+(class|void|int)", r"SELECT\s+", r"```", r"import\s+", r"package\s+"]
        return any(re.search(p, text) for p in indicators)

    def finetune(self, data: Union[str, Dataset, List[str]], epochs: int = 5, learning_rate: float = 1e-4, **kwargs: Any) -> Dict[str, Any]:
        logger.info(f"Fine-tuning model '{self.name}' for {epochs} epochs...")
        return self.train(data=data, method="finetune", epochs=epochs, learning_rate=learning_rate, **kwargs)

    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        ds = dataset or Dataset.from_texts(self.knowledge_base.facts)
        evaluator = Evaluator(self)
        eval_results = evaluator.evaluate(ds)
        if self.learning_validator:
            vs = self.learning_validator.get_validation_summary()
            eval_results.update({"learning_accuracy": vs.get("average_accuracy", 0), "comprehension": vs.get("average_comprehension", 0), "knowledge_retention": vs.get("average_retention", 0), "language_understanding": vs.get("average_language_understanding", 0)})
        return eval_results

    def benchmark(self, prompt: str = "Who are you?", num_runs: int = 20) -> Dict[str, Any]:
        benchmarker = Benchmarker(self)
        return benchmarker.benchmark(prompt=prompt, num_runs=num_runs)

    def save(self, filepath: Optional[str] = None) -> str:
        target_path = filepath or f"{self.name.lower()}{DEFAULT_MODEL_EXTENSION}"
        return ModelExporter.export(self, target_path, format="ct")

    def export(self, output_path: str, format: Optional[str] = None) -> str:
        return ModelExporter.export(self, output_path, format=format)

    def load(self, filepath: str) -> "Model":
        if not os.path.exists(filepath):
            raise ModelError(f"Model file not found: '{filepath}'")
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".ct" or filepath.endswith(".ct"):
            parsed = CarvusTrainParser.parse_file(filepath)
            if parsed.model_config.get("Name"):
                self.name = parsed.model_config.get("Name")
            self.knowledge_base.add_facts(parsed.knowledge)
            logger.info(f"Loaded {len(parsed.knowledge)} knowledge facts from '{filepath}'")
        logger.info(f"Successfully loaded model state from '{filepath}'")
        return self

    def serve(self, port: int = 8000, host: str = "0.0.0.0") -> None:
        """Start a REST API server for model inference.

        Args:
            port: Server port number (default: 8000).
            host: Server host address (default: 0.0.0.0).

        Provides:
            POST /  - Ask a question: {"prompt": "your question"}
            GET /   - Get model status info
        """
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import json

        model_ref = self  # Capture outer self for closure

        logger.info(f"Starting CarvusTrain Model Server on {host}:{port}...")

        class ModelHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
                data = json.loads(body) if body else {}
                prompt = data.get("prompt", "")
                response = model_ref.ask(prompt) if prompt else "No prompt provided."
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"response": response, "model": model_ref.name, "mode": model_ref.mode}).encode())

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"name": model_ref.name, "architecture": model_ref.architecture, "mode": model_ref.mode, "knowledge_facts": len(model_ref.knowledge_base), "learning_accuracy": model_ref.knowledge_base.get_average_accuracy(), "status": "running"}).encode())

            def log_message(self, format, *args):
                pass

        try:
            server = HTTPServer((host, port), ModelHandler)
            logger.success(f"Model server running at http://{host}:{port}")
            print(f"\n  CarvusTrain Server running at http://{host}:{port}")
            print('  POST / - ask a question ({"prompt": "your question"})')
            print('  GET / - model info\n')
            server.serve_forever()
        except OSError as e:
            logger.error(f"Could not start server on port {port}: {e}")

    def summary(self) -> None:
        print("=" * 60)
        print(f"CarvusTrain Model Summary: {self.name}")
        print("=" * 60)
        print(f"Architecture:       {self.architecture}")
        print(f"Parameters:         {self.parameters}")
        print(f"Mode:               {self._mode}")
        print(f"Memory Enabled:     {self._memory_enabled}")
        print(f"Plugins:            {len(self._plugins)} installed")
        print(f"Model Type:         {self.config.model.model_type}")
        print(f"Vocab Size:         {self.config.model.vocab_size}")
        print(f"Embedding Dim:      {self.config.model.embedding_dim}")
        print(f"Hidden Dim:         {self.config.model.hidden_dim}")
        print(f"Num Layers:         {self.config.model.num_layers}")
        print(f"Num Heads:          {self.config.model.num_heads}")
        print(f"Knowledge Facts:    {len(self.knowledge_base)}")
        print(f"Code Languages:     {', '.join(self.knowledge_base.get_programming_languages()) or 'None'}")
        print(f"Learning Acc:       {self.knowledge_base.get_average_accuracy():.4f}")
        if self._personality:
            p = self._personality
            print(f"Personality:        style={p.get('style', 'N/A')}, humor={p.get('humor', 0)}, creativity={p.get('creativity', 0)}")
        print("=" * 60)

    def statistics(self) -> Dict[str, Any]:
        sys_info = get_system_info()
        return {"model_name": self.name, "architecture": self.architecture, "mode": self._mode, "memory_enabled": self._memory_enabled, "vocab_size": len(self.tokenizer.vocab), "knowledge_facts_count": len(self.knowledge_base), "code_languages": self.knowledge_base.get_programming_languages(), "learning_accuracy": self.knowledge_base.get_average_accuracy(), "plugins": list(self._plugins.keys()), "system_info": sys_info}

    def memory(self) -> Dict[str, Any]:
        return {"stored_facts": len(self.knowledge_base), "context_window_max_tokens": self.inference_engine.chat_session.context.max_tokens, "vocab_token_count": len(self.tokenizer.vocab), "code_languages": self.knowledge_base.get_programming_languages(), "learning_status": self.knowledge_base.get_learning_status(), "conversation_history_length": len(self._conversation_history) if self._memory_enabled else 0}

    def to_dict(self) -> Dict[str, Any]:
        d = {"name": self.name, "architecture": self.architecture, "mode": self._mode, "config": self.config.to_dict(), "knowledge": self.knowledge_base.facts, "vocab": self.tokenizer.vocab.to_dict() if hasattr(self.tokenizer.vocab, "to_dict") else {}, "learning_metrics": {"average_accuracy": self.knowledge_base.get_average_accuracy(), "average_learning_score": self.knowledge_base.get_average_learning_score(), "code_languages": self.knowledge_base.get_programming_languages()}}
        if self._personality:
            d["personality"] = self._personality
        return d


class ChatModel(Model):
    """Specialized Model optimized for multi-turn interactive chat."""
    def __init__(self, name: str = "CarvusChat", **kwargs: Any) -> None:
        kwargs.pop("mode", None)  # Remove mode if passed to avoid conflict
        super().__init__(name=name, mode="general", **kwargs)


class TextModel(Model):
    """Specialized Model optimized for text generation."""
    def __init__(self, name: str = "CarvusText", **kwargs: Any) -> None:
        kwargs.pop("mode", None)
        super().__init__(name=name, mode="general", **kwargs)


class LanguageModel(Model):
    """Specialized Model optimized for language modeling."""
    def __init__(self, name: str = "CarvusLM", **kwargs: Any) -> None:
        kwargs.pop("mode", None)
        super().__init__(name=name, mode="general", **kwargs)


class CustomModel(Model):
    """Customizable Model class for user-defined architectures."""
    def __init__(self, name: str = "CarvusCustom", **kwargs: Any) -> None:
        kwargs.pop("mode", None)
        super().__init__(name=name, mode="general", **kwargs)


class AgentModel(Model):
    """Goal-oriented AI Agent that can be trained for specific tasks."""

    def __init__(self, name: str = "CarvusAgent", goal: str = "software engineer", **kwargs: Any) -> None:
        self.goal = goal
        self._sub_agents: Dict[str, AgentModel] = {}
        super().__init__(name=name, mode="programmer", **kwargs)

    def train_agent(self, goal: Optional[str] = None, data: Optional[Union[str, Dataset, List[str]]] = None) -> Dict[str, Any]:
        if goal:
            self.goal = goal
        logger.info(f"Training agent '{self.name}' for goal: {self.goal}")
        return self.auto_train(data=data)

    def add_sub_agent(self, name: str, agent: "AgentModel") -> None:
        self._sub_agents[name] = agent
        logger.info(f"Sub-agent '{name}' added to '{self.name}'")

    def orchestrate(self, task: str) -> Dict[str, str]:
        results = {"planner": f"Planning: {task}", "executor": self.ask(task)}
        for name, agent in self._sub_agents.items():
            results[name] = agent.ask(task)
        return results
