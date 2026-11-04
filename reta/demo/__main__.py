"""Run `python -m reta.demo --help` for the local demo server."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="ReTA interactive example demo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--icd2ccs", required=True)
    parser.add_argument("--ccs-hierarchy", required=True)
    parser.add_argument("--descriptions", required=True, help="Canonical ICD:/CCS:/KG ID -> name JSON")
    parser.add_argument("--results", help="Optional aggregate experiment results JSON")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import torch
    import uvicorn
    from .app import DemoModel, create_app
    torch.set_num_threads(2)
    app = create_app(lambda: DemoModel(args.config, args.checkpoint, args.icd2ccs,
                                      args.ccs_hierarchy, args.descriptions, args.device, args.results))
    uvicorn.run(app, host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
