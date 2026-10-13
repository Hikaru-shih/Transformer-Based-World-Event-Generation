"""Publish frozen experiment results and traceable run records as English Markdown."""
import json
from pathlib import Path
import shutil
from protocol import ROOT
from run_conditioning import run_path, METHODS, SEEDS

OUT = ROOT/'results/research-conditioning'
DEST = ROOT/'docs/conditioning_results'

def main():
    result = json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
    DEST.mkdir(parents=True,exist_ok=True)
    shutil.copy2(OUT/'comparison.json',DEST/'comparison.json')
    shutil.copy2(OUT/'plan.json',DEST/'plan.json')
    lines = ['# Conditioning Comparison Results','',
        'Four model settings were evaluated across seeds 42, 43, and 44 on the frozen 192-example AI-reviewed synthetic benchmark. Each strict compositional test subset contains 24 examples. All single field values are seen in training; Location/NPC pairs are held out. Results below are mean ± sample standard deviation across three runs.', '',
        '## Primary Metrics','',
        '| Method | Parameters | NPC explicit accuracy | Player explicit accuracy | Token overlap F1 |',
        '|---|---:|---:|---:|---:|']
    def format_value(method,key):
        value = result['summary'][method][key]
        return f"{value['mean']:.3f} ± {value['sd']:.3f}"
    for method in (*METHODS,'retrieval'):
        parameters = next((r.get('parameters') for r in result['runs'] if r['method']==method),None)
        lines.append(f"| {method} | {parameters if parameters is not None else 'N/A'} | {format_value(method,'npc_explicit_accuracy')} | {format_value(method,'player_explicit_accuracy')} | {format_value(method,'token_overlap_f1')} |")
    lines.extend(['','## Supplementary Context Markers','',
                  '| Method | Location | Time | Weather |','|---|---:|---:|---:|'])
    for method in (*METHODS,'retrieval'):
        lines.append(f"| {method} | {format_value(method,'location_context_marker_accuracy')} | {format_value(method,'time_context_marker_accuracy')} | {format_value(method,'weather_context_marker_accuracy')} |")
    lines.extend(['','These checks require explicit vocabulary or generator-specific phrases. They do not establish causal plausibility, interestingness, or full semantic consistency. Missing markers or reasonable paraphrases may fail. High overlap F1 does not compensate for an incorrect actor.', '',
                  '## Per-Run Checkpoints and Conditioning Diagnostics','',
                  '| Method | Seed | Selected epoch | Conditioning NLL gap | NLL unit |','|---|---:|---:|---:|---|'])
    for method in METHODS:
        for seed in SEEDS:
            path = run_path(method,seed)
            folder = DEST/f'{method}-{seed}'
            folder.mkdir(exist_ok=True)
            for name in ('manifest.json','history.json'):
                shutil.copy2(path/name,folder/name)
            history = json.loads((path/'history.json').read_text(encoding='utf-8'))
            key = 'validation_word_nll' if method.startswith('words-') else 'validation_event_nll'
            best = min(history,key=lambda row:row[key])['epoch']
            record = next(r for r in result['runs'] if r['method']==method and r['seed']==seed)
            unit = 'word' if method.startswith('words-') else 'character'
            lines.append(f"| {method} | {seed} | {best} | {record['conditioning_nll_gap']:.4f} | {unit} |")
            if method.startswith('words-'):
                for name in ('validation_metrics.json','validation_predictions.csv','test_metrics.json','test_predictions.csv'):
                    shutil.copy2(path/name,folder/name)
            else:
                source = OUT/f'{method}-{seed}-evaluation'
                for name in ('metrics.json','human_review.csv','system_key.json'):
                    shutil.copy2(source/name,folder/name)
    lines.extend(['','NLL units differ between tokenizers. Compare conditioning gaps within a method; do not compare character and word magnitudes directly. A positive gap indicates a likelihood advantage for the correct state in this diagnostic, not proof of general narrative quality.', '',
                  '## Interpretation and Limits','',
                  '- Word-flat and word-slots share architecture, capacity, optimizer, and seed choices. Their comparison examines memory construction. A character-to-word comparison also changes tokenization, architecture, and capacity, so it cannot isolate a field-encoding effect.',
                  '- The dataset shares rule templates across subsets and has AI review only. It has zero independent human-reviewed targets and no completed human ratings of generated events.',
                  '- Earlier test results on this benchmark were inspected. The current settings were frozen before the multi-seed test phase, but this remains an exploratory synthetic benchmark rather than a pristine final test.',
                  '- Split membership and initialization both vary with seed. Three-run standard deviations summarize these runs, not statistical significance or initialization-only variance.',
                  '- Each state has one target. Exact-match differences can reflect alternate valid phrasing or actions; no reference-overlap metric alone proves semantic quality.', '',
                  '## Reproduction and Evidence','',
                  'See [Conditioning Experiment Protocol](CONDITIONING_EXPERIMENT.md) for the frozen setup, runner commands, and generation demo. The complete comparison, frozen plan, run manifests, histories, predictions, and unscored review sheets are retained in `docs/conditioning_results/`. Checkpoints and logs remain in the original `results/` run directories. No human review scores have been filled automatically.', ''])
    interpretation = ROOT/'docs/CONDITIONING_INTERPRETATION.md'
    if interpretation.exists():
        lines.extend(['',interpretation.read_text(encoding='utf-8')])
    (ROOT/'docs/CONDITIONING_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:14]))

if __name__ == '__main__':
    main()
