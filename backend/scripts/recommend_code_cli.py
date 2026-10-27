"""Run: python cli.py --help"""
import argparse
import json
from pathlib import Path
from recommender import DEFAULT_MODEL, recommend, safe_path


def main():
    parser = argparse.ArgumentParser(description='Generate review-only Nemotron code proposals.')
    parser.add_argument('--root', type=Path, required=True, help='User training project root')
    parser.add_argument('--files', nargs='+', required=True, help='Explicit relative file paths')
    parser.add_argument('--context', type=Path, help='JSON metadata file; not treated as editable code')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--max-tokens', type=int, default=16000)
    parser.add_argument('--out', type=Path, default=Path('recommendations'))
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)
        output = args.out.resolve()
        # Proposals must not land inside the input project.
        if output.is_relative_to(root):
            raise ValueError('--out must be outside --root.')
        if output.exists():
            raise ValueError('Output already exists; choose a new --out directory.')
        files = {}
        total = 0
        for name in args.files:
            name = safe_path(name)
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f'File escapes project or is not a regular file: {name}')
            total += path.stat().st_size
            if total > 250_000:
                raise ValueError('Selected files exceed 250 KB.')
            files[name] = path.read_text(encoding='utf-8')
        context = json.loads(args.context.read_text()) if args.context else {}
        if not isinstance(context, dict):
            raise ValueError('Context must be a JSON object.')
        result = recommend(prompt=args.prompt, files=files, context=context,
                           model=args.model, max_tokens=args.max_tokens)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        (output / 'changes.diff').write_text(result['validation']['diff'], encoding='utf-8')
        review = result['review']
        report = [review['explanation'], '\nValidation: static checks only; not runtime tested.']
        for item in review['recommendations']:
            report += ['\n## ' + item['title'], item['explanation']]
            for key in ('evidence', 'affected_files', 'tradeoffs', 'verification_steps'):
                report += ['\n' + key.replace('_', ' ').title() + ':']
                report += ['- ' + value for value in item[key]]
        for key in ('assumptions', 'missing_information'):
            report += ['\n## ' + key.replace('_', ' ').title()]
            report += ['- ' + value for value in review[key]]
        (output / 'report.md').write_text('\n'.join(report), encoding='utf-8')
        for change in review['changed_files']:
            target = output / 'proposed' / change['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(change['content'], encoding='utf-8')
        print(f'Review and proposed files: {output}')
    except Exception as exc:
        # Avoid printing raw SDK responses, credentials, or submitted code.
        from openai import APIError
        if isinstance(exc, APIError):
            parser.exit(1, f'API request failed ({type(exc).__name__}); check credentials, quota and model availability.\n')
        parser.exit(1, f'Review failed: {type(exc).__name__}: {exc}\n')

if __name__ == '__main__':
    main()
