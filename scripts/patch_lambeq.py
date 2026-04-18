"""
Monkey-patch for lambeq's broken model downloader.

The model server at qnlp.cambridgequantum.com has been offline since
late 2025 (Cambridge Quantum rebranded to Quantinuum). This patch
modifies the `_prepare_model_artifacts` method so that if the model
can't be downloaded and no cache exists, it prints clear instructions
on how to obtain the model manually.

Usage — add these 2 lines at the TOP of any script that uses BobcatParser:

    import sys; sys.path.insert(0, '.')
    import scripts.patch_lambeq   # noqa: F401

    from lambeq import BobcatParser
    parser = BobcatParser(verbose='suppress')
"""

from pathlib import Path

import lambeq.text2diagram.model_based_reader.model_downloader as md

# Store the original __init__
_original_init = md.ModelDownloader.__init__


def _patched_init(self, model_name, cache_dir=None):
    """Patched __init__ that doesn't crash when the server is unreachable."""
    if model_name not in md.MODELS:
        raise ValueError(f'Invalid model name: {model_name!r}')

    self.model = model_name
    self.model_dir = self.get_dir(cache_dir)
    self.model_url = self.get_url()

    try:
        self.remote_version = self.get_latest_remote_version()
    except Exception:
        # Server unreachable — check if we have a local copy
        self.remote_version = None
        self.version_retrieval_error = md.ModelDownloaderError(
            f"Cannot reach model server. If you have a local copy of the "
            f"{model_name} model, place it in: {self.model_dir}\n"
            f"Otherwise, ask someone who has the model to share the "
            f"~/.cache/lambeq/{model_name}/ directory with you."
        )

        local_version = self.get_local_model_version()
        if local_version is not None:
            # We have a cached model — pretend the server returned the
            # same version so model_is_stale() returns False
            self.remote_version = local_version
            print(f"[patch] Server offline, using cached {model_name} "
                  f"model (v{local_version})")


md.ModelDownloader.__init__ = _patched_init

print("[patch_lambeq] ✓ Patched model downloader to handle offline server.")
