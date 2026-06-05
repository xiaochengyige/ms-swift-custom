# Copyright (c) ModelScope Contributors. All rights reserved.
"""Self-check for the OneVL non-invasive plugin (no model weights required).

It mimics what ``--external_plugins onevl/register.py`` does and then
verifies every non-invasive registration / injection took effect:

  * model_type  'qwen3_vl_latent_cot'                registered
  * template    'qwen3_vl_latent_cot'                registered
  * template    'qwen3_vl_latent_cot_explain'        registered
  * loss_map contains 'latent_cot'
  * RowPreprocessor.standard_keys contains the two extra columns
  * TunerMixin.prepare_model has been monkey-patched

This module does NOT touch ``sys.path``. Import resolution is provided entirely
via ``PYTHONPATH``, which must contain BOTH the ``onevl/`` directory (so
``register`` / ``onevl_plugin`` are importable) and ``ms-swift-4.2.0/`` (so
``swift`` and its deps are importable). The easiest way is the wrapper:

    bash onevl/scripts/selfcheck.sh

or set it manually, e.g. from OneVL/custom:

    PYTHONPATH="onevl:ms-swift-4.2.0:${PYTHONPATH}" python onevl/selfcheck.py
"""


def main() -> int:
    try:
        import register  # noqa: F401  -> triggers all OneVL registrations
    except Exception as e:  # pragma: no cover - environment dependent
        print(f'[FAIL] importing register.py failed: {e!r}')
        print('       Ensure PYTHONPATH contains BOTH the onevl/ dir (for register.py /')
        print('       onevl_plugin) and ms-swift-4.2.0/ (for swift + torch/transformers).')
        print('       Tip: just run `bash onevl/scripts/selfcheck.sh`.')
        return 2

    from swift.model import MODEL_MAPPING
    from swift.template import TEMPLATE_MAPPING
    from swift.loss.mapping import loss_map
    from swift.dataset.preprocessor.core import RowPreprocessor
    from swift.pipelines.train.tuner import TunerMixin

    checks = [
        ("model_type 'qwen3_vl_latent_cot' registered", 'qwen3_vl_latent_cot' in MODEL_MAPPING),
        ("template 'qwen3_vl_latent_cot' registered", 'qwen3_vl_latent_cot' in TEMPLATE_MAPPING),
        ("template 'qwen3_vl_latent_cot_explain' registered", 'qwen3_vl_latent_cot_explain' in TEMPLATE_MAPPING),
        ("loss_map contains 'latent_cot'", 'latent_cot' in loss_map),
        ("standard_keys contains 'think_steps'", 'think_steps' in RowPreprocessor.standard_keys),
        ("standard_keys contains 'future_image_tokens'", 'future_image_tokens' in RowPreprocessor.standard_keys),
        ('TunerMixin.prepare_model monkey-patched', getattr(TunerMixin, '_onevl_latent_cot_patched', False)),
    ]

    ok = True
    for name, passed in checks:
        print(f'[{"PASS" if passed else "FAIL"}] {name}')
        ok = ok and bool(passed)

    print('\nAll checks passed.' if ok else '\nSome checks FAILED.')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
