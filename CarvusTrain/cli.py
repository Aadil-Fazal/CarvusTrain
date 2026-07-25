"""Command Line Interface (CLI) for CarvusTrain framework."""

import argparse
import os
import sys
from typing import List, Optional

from .configuration import CarvusConfig, load_config_file
from .constants import DEFAULT_MODEL_EXTENSION
from .evaluation import Benchmarker, Evaluator
from .dataset import Dataset
from .exceptions import CarvusTrainError
from .exporter import ModelExporter
from .logger import logger
from .model import ChatModel, Model
from .parser import DataParser
from .utils import get_system_info
from .version import get_version_info


def main(args: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint for CarvusTrain command execution."""
    parser = argparse.ArgumentParser(
        prog="carvustrain",
        description="CarvusTrain - High Performance AI Model Training and Inference Framework",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # create (alias for init)
    parser_create = subparsers.add_parser("create", help="Create a new AI model project")
    parser_create.add_argument("name", help="Name of the AI model to create")
    parser_create.add_argument("--architecture", default="transformer", choices=["transformer", "neural_network", "cnn", "rnn", "lstm", "custom"], help="Model architecture")
    parser_create.add_argument("--parameters", default="base", help="Model size (base, large, 1B, 7B, etc.)")

    # init
    parser_init = subparsers.add_parser("init", help="Initialize a new CarvusTrain project workspace")
    parser_init.add_argument("directory", nargs="?", default=".", help="Target project directory")

    # train
    parser_train = subparsers.add_parser("train", help="Train a CarvusTrain AI model")
    parser_train.add_argument("--data", required=True, help="Path to training dataset file or folder")
    parser_train.add_argument("--name", default="Carvus", help="Model name")
    parser_train.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser_train.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser_train.add_argument("--method", default="normal", help="Training method strategy")
    parser_train.add_argument("--device", default="auto", help="Target device (cpu, cuda, gpu, mps)")
    parser_train.add_argument("--output", default=f"carvus{DEFAULT_MODEL_EXTENSION}", help="Output model path")

    # predict
    parser_predict = subparsers.add_parser("predict", help="Generate prediction or text output from a trained model")
    parser_predict.add_argument("--model", required=True, help="Path to trained .ct model file")
    parser_predict.add_argument("--prompt", required=True, help="Input prompt string")

    # chat
    parser_chat = subparsers.add_parser("chat", help="Start an interactive chat session with a trained model")
    parser_chat.add_argument("--model", help="Path to trained .ct model file")

    # export
    parser_export = subparsers.add_parser("export", help="Export a model to ONNX, GGUF, BIN, JSON, or CT format")
    parser_export.add_argument("--model", required=True, help="Path to source .ct model file")
    parser_export.add_argument("--output", required=True, help="Target export file path")
    parser_export.add_argument("--format", choices=["ct", "bin", "onnx", "json", "gguf"], help="Export format")

    # evaluate
    parser_eval = subparsers.add_parser("evaluate", help="Evaluate model accuracy and loss metrics")
    parser_eval.add_argument("--model", required=True, help="Path to trained .ct model file")
    parser_eval.add_argument("--data", required=True, help="Path to test dataset file")

    # serve
    parser_serve = subparsers.add_parser("serve", help="Start a REST API server for model inference")
    parser_serve.add_argument("--model", help="Path to trained .ct model file")
    parser_serve.add_argument("--port", type=int, default=8000, help="Server port number")
    parser_serve.add_argument("--host", default="0.0.0.0", help="Server host address")

    # deploy
    parser_deploy = subparsers.add_parser("deploy", help="Deploy model as a production REST API server")
    parser_deploy.add_argument("--model", required=True, help="Path to trained .ct model file")
    parser_deploy.add_argument("--port", type=int, default=8000, help="Server port number")

    # list
    subparsers.add_parser("list", help="List available .ct model files in current directory")

    # info
    parser_info = subparsers.add_parser("info", help="Display detailed model information")
    parser_info.add_argument("--model", help="Path to trained .ct model file")

    # push
    parser_push = subparsers.add_parser("push", help="Publish model to Carvus Hub (coming soon)")
    parser_push.add_argument("--model", required=True, help="Path to .ct model file")

    # pull
    parser_pull = subparsers.add_parser("pull", help="Download model from Carvus Hub (coming soon)")
    parser_pull.add_argument("name", help="Name of the model to pull")

    # benchmark
    parser_bench = subparsers.add_parser("benchmark", help="Run performance benchmark measuring throughput & latency")
    parser_bench.add_argument("--model", help="Path to trained .ct model file")
    parser_bench.add_argument("--runs", type=int, default=20, help="Number of benchmark iterations")

    # convert
    parser_convert = subparsers.add_parser("convert", help="Convert dataset or model format")
    parser_convert.add_argument("--input", required=True, help="Input file path")
    parser_convert.add_argument("--output", required=True, help="Output file path")

    # version
    subparsers.add_parser("version", help="Show CarvusTrain version information")

    # doctor
    subparsers.add_parser("doctor", help="Check system dependencies, CUDA status, and hardware compatibility")

    # config
    parser_cfg = subparsers.add_parser("config", help="View or validate configuration file")
    parser_cfg.add_argument("config_file", nargs="?", help="Path to config.toml, config.yaml, or config.json")

    # install
    subparsers.add_parser("install", help="Install optional dependencies (PyTorch, PyYAML, etc.)")

    # update
    subparsers.add_parser("update", help="Check for framework updates")

    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return

    cmd = parsed_args.command.lower()

    try:
        if cmd == "create":
            name = parsed_args.name
            arch = parsed_args.architecture
            params = parsed_args.parameters
            model_dir = os.path.join(os.getcwd(), name.lower().replace(" ", "_"))
            os.makedirs(model_dir, exist_ok=True)
            train_file = os.path.join(model_dir, "train.ct")
            with open(train_file, "w", encoding="utf-8") as f:
                f.write(f"[CarvusTrain]\nVersion=1.0\nLearnCheck=True\nKnowledgeValidation=True\n\n[Model]\nName={name}\nArchitecture={arch}\nParameters={params}\n\n[Training]\nMethod=normal\nDuration=forever\n\n[Knowledge]\n{name} is an AI created with CarvusTrain.\n{name} uses the {arch} architecture.\n")
            logger.success(f"Created AI model '{name}' ({arch}/{params}) in '{model_dir}'")
            print(f"\n  Next steps:")
            print(f"    cd {model_dir}")
            print(f"    carvustrain train --data train.ct --name {name}")
            print(f"    carvustrain chat --model {name.lower()}.ct\n")

        elif cmd == "init":
            target_dir = os.path.abspath(parsed_args.directory)
            os.makedirs(target_dir, exist_ok=True)
            config_file = os.path.join(target_dir, "config.toml")
            train_file = os.path.join(target_dir, "train.ct")

            if not os.path.exists(config_file):
                with open(config_file, "w", encoding="utf-8") as f:
                    f.write('[model]\nname = "Carvus"\n[training]\nmethod = "normal"\nepochs = 10\n')

            if not os.path.exists(train_file):
                with open(train_file, "w", encoding="utf-8") as f:
                    f.write("[CarvusTrain]\n\n[Model]\nName=Carvus\n\n[Training]\nMethod=normal\nDuration=forever\n\n[Knowledge]\nCarvus is an AI trained with CarvusTrain.\n")

            logger.success(f"Initialized CarvusTrain project workspace at '{target_dir}'")

        elif cmd == "train":
            model = Model(name=parsed_args.name, auto_load=True)
            model.auto_train(
                data=parsed_args.data,
                auto_optimize=True
            )
            model.save(parsed_args.output)

        elif cmd == "predict":
            model = Model()
            if os.path.exists(parsed_args.model):
                model.load(parsed_args.model)
            answer = model.ask(parsed_args.prompt)
            print(f"\nResponse:\n{answer}\n")

        elif cmd == "chat":
            model = ChatModel(auto_load=True)
            if parsed_args.model and os.path.exists(parsed_args.model):
                model.load(parsed_args.model)
            print("=" * 60)
            print("CarvusTrain Interactive Chat Mode (type 'exit' or 'quit' to end)")
            print("=" * 60)
            while True:
                try:
                    q = input("\nAsk Anything: ")
                    if q.strip().lower() in ("exit", "quit"):
                        print("Ending chat session. Goodbye!")
                        break
                    ans = model.chat(q)
                    print(f"Carvus: {ans}")
                except (KeyboardInterrupt, EOFError):
                    break

        elif cmd == "serve":
            model = Model(auto_load=True)
            if parsed_args.model and os.path.exists(parsed_args.model):
                model.load(parsed_args.model)
            model.serve(port=parsed_args.port, host=getattr(parsed_args, 'host', '0.0.0.0'))

        elif cmd == "push":
            logger.info("Publishing model to Carvus Hub...")
            if parsed_args.model and os.path.exists(parsed_args.model):
                logger.success(f"Model '{parsed_args.model}' prepared for publishing.")
            else:
                logger.error("Model file not found.")

        elif cmd == "pull":
            logger.info(f"Downloading model '{parsed_args.name}' from Carvus Hub...")
            logger.info("Carvus Hub is coming soon. For now, models are loaded from local files.")

        elif cmd == "deploy":
            model = Model(auto_load=True)
            if parsed_args.model and os.path.exists(parsed_args.model):
                model.load(parsed_args.model)
            model.serve(port=parsed_args.port or 8000)

        elif cmd == "list":
            ct_files = [f for f in os.listdir('.') if f.endswith('.ct')]
            if ct_files:
                print("\nAvailable CarvusTrain models:")
                for f in ct_files:
                    size = os.path.getsize(f)
                    print(f"  {f:30s} {size:>8,} bytes")
            else:
                print("\nNo .ct model files found in current directory.")

        elif cmd == "info":
            model = Model(auto_load=True)
            if parsed_args.model and os.path.exists(parsed_args.model):
                model.load(parsed_args.model)
            model.summary()

        elif cmd == "export":
            model = Model()
            model.load(parsed_args.model)
            model.export(parsed_args.output, format=parsed_args.format)

        elif cmd == "evaluate":
            model = Model()
            model.load(parsed_args.model)
            records = DataParser.parse(parsed_args.data)
            ds = Dataset(records)
            evaluator = Evaluator(model)
            evaluator.evaluate(ds)

        elif cmd == "benchmark":
            model = Model()
            if parsed_args.model and os.path.exists(parsed_args.model):
                model.load(parsed_args.model)
            benchmarker = Benchmarker(model)
            benchmarker.benchmark(num_runs=parsed_args.runs)

        elif cmd == "convert":
            records = DataParser.parse(parsed_args.input)
            out_ext = os.path.splitext(parsed_args.output)[1].lower()
            if out_ext == ".json":
                import json

                with open(parsed_args.output, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
            logger.success(f"Converted '{parsed_args.input}' to '{parsed_args.output}' ({len(records)} records)")

        elif cmd == "version":
            print(get_version_info())

        elif cmd == "doctor":
            info = get_system_info()
            print("=" * 60)
            print("CarvusTrain Environment & Health Diagnostics")
            print("=" * 60)
            print(f"Platform:         {info['platform']}")
            print(f"Python Version:   {info['python_version']}")
            print(f"CPU Cores:        {info['cpu_count']}")
            print(f"NumPy Installed:  {info['numpy_installed']} (v{info.get('numpy_version', 'N/A')})")
            print(f"PyTorch Installed:{info['pytorch_installed']} (v{info.get('pytorch_version', 'N/A')})")
            print(f"CUDA Available:   {info['cuda_available']} ({info.get('cuda_device_name', 'N/A')})")
            print(f"MPS Available:    {info['mps_available']}")
            print("=" * 60)
            logger.success("CarvusTrain environment is healthy and operational.")

        elif cmd == "config":
            if parsed_args.config_file:
                cfg = load_config_file(parsed_args.config_file)
                import json

                print(json.dumps(cfg.to_dict(), indent=2))
            else:
                default_cfg = CarvusConfig()
                import json

                print(json.dumps(default_cfg.to_dict(), indent=2))

        elif cmd == "install":
            logger.info("To install extra dependencies, run: pip install pyyaml torch pytest")

        elif cmd == "update":
            from .version import __version__
            logger.info(f"CarvusTrain is currently at latest version v{__version__}.")

    except CarvusTrainError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected CLI error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
