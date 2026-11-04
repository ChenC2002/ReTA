"""Open the public viewer or serve a complete model bundle locally."""
import argparse
import json
import webbrowser

DEMO_URL = 'https://chenc2002.github.io/ReTA/demo/'


def main(argv=None):
    parser = argparse.ArgumentParser(description='ReTA interactive demo')
    parser.add_argument('--bundle', help='Extracted V2.0 model package directory')
    parser.add_argument('--verify', action='store_true', help='Replay all bundled examples and exit')
    parser.add_argument('--no-open', action='store_true', help='Print the online viewer URL without opening it')
    for name in ('config', 'checkpoint', 'icd2ccs', 'ccs-hierarchy', 'descriptions', 'results'):
        parser.add_argument('--'+name)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    legacy = (args.config, args.checkpoint, args.icd2ccs, args.ccs_hierarchy, args.descriptions)
    if args.bundle and any(legacy):
        parser.error('Use --bundle or individual model paths, not both.')
    if args.verify and not args.bundle:
        parser.error('--verify requires --bundle.')
    if not args.bundle and not any(legacy):
        print(DEMO_URL)
        if not args.no_open:
            webbrowser.open(DEMO_URL)
        return 0
    if not args.bundle and not all(legacy):
        parser.error('Supply --bundle or all five model resource paths.')
    import torch
    torch.set_num_threads(2)
    if args.verify:
        from .bundle import verify_bundle
        print(json.dumps(verify_bundle(args.bundle, args.device), indent=2))
        return 0
    import uvicorn
    from .app import DemoModel, create_app
    if args.bundle:
        from .bundle import load_bundle
        factory = lambda: load_bundle(args.bundle, args.device)
    else:
        factory = lambda: DemoModel(*legacy, device=args.device, results=args.results)
    print(f'ReTA demo: http://127.0.0.1:{args.port}/', flush=True)
    uvicorn.run(create_app(factory), host='127.0.0.1', port=args.port, access_log=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
