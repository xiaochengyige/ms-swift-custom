# Copyright (c) ModelScope Contributors. All rights reserved.
"""OneVL latent-CoT non-invasive plugin package for ms-swift 4.2.0.

This package reproduces the OneVL training/inference extensions of ms-swift
without modifying any swift source file.  Importing the submodules
(``model`` / ``template`` / ``infer``) triggers ``register_model`` /
``register_template`` as a side effect; ``loss`` / ``latent_cot`` are imported
on demand.

Do NOT import the registering submodules here -- registration is driven
explicitly by the ``register.py`` entry file (the ``--external_plugins`` target)
to keep a single, well-defined registration point.
"""

__all__ = []
