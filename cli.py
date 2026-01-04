import argparse
import yaml
from src import ingest, preprocess, build_dataset

def load_config(path="config/default.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Marathi HTR data tool")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ingest")
    subparsers.add_parser("preprocess")
    subparsers.add_parser("build-dataset")

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "ingest":
        ingest.run(cfg)
    elif args.command == "preprocess":
        preprocess.run(cfg)
    elif args.command == "build-dataset":
        build_dataset.run(cfg)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
