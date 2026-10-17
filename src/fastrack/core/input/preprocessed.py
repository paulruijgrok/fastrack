"""Frame source that applies illumination / low-SNR preprocessing on read.

``PreprocessedSource`` wraps any other :class:`FrameSource` and returns
corrected, full-range frames instead of the raw ones, so every consumer of the
frame-source seam (the gliding pipeline, its multiprocessing workers, the
overlay movie, ``Motility``) gets preprocessed pixels without knowing about it.

The correction (``fastrack.preprocess``) needs a *movie-wide* model -- the
temporal-median background, the illumination gain field and the stretch
ceiling.  Fitting it needs several frames at once, which the strictly per-frame
detection flow never holds; this source resolves that by fitting **once** and
persisting the model as a small ``.npz`` next to the detection cache
(``model_path``).  The parent process calls :meth:`ensure_model` before
spawning workers; each worker then rebuilds the source from its descriptor and
loads the saved model rather than refitting.  Long movies that don't fit in
memory use ``model_frames=N`` to fit on N evenly spaced frames.
"""
import os

from .base import FrameSource, open_source, register_source, to_pipeline_image


@register_source("preprocessed")
class PreprocessedSource(FrameSource):
    def __init__(self, inner, params, model_path=None):
        """``inner``: a FrameSource or its descriptor; ``params``: recipe dict
        (see :data:`fastrack.preprocess.DEFAULT_PARAMS`); ``model_path``: where
        the fitted model is persisted (``None`` -> fit in-process, no cache)."""
        self.inner = inner if isinstance(inner, FrameSource) else open_source(inner)
        self.params = dict(params)
        self.model_path = model_path
        self._model = None

    # ----- model ----------------------------------------------------------- #
    def ensure_model(self, force=False):
        """Fit (or load) the model; save it to ``model_path`` when set."""
        from ...preprocess import PreprocessModel, fit_model_from_reader

        if self._model is not None and not force:
            return self._model
        if self.model_path and os.path.isfile(self.model_path) and not force:
            self._model = PreprocessModel.load(self.model_path)
            if self._model.noise_sd is not None:
                return self._model
            # A model saved before noise_sd existed: refit once (same pixels,
            # plus the noise estimate) and overwrite it below.
        self._model = fit_model_from_reader(
            self.inner.read_raw, self.inner.frame_numbers(), **self.params)
        if self.model_path:
            os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
            self._model.save(self.model_path)
        return self._model

    @property
    def model(self):
        return self.ensure_model()

    # ----- FrameSource API ------------------------------------------------- #
    @property
    def count(self):
        return self.inner.count

    def frame_numbers(self):
        return self.inner.frame_numbers()

    def elapsed_times(self):
        return self.inner.elapsed_times()

    def read(self, frame_no):
        # 8-bit corrected frame, then the same x257 the legacy read applies, so a
        # preprocessed frame is exactly what the pipeline would see reading the
        # tools/preprocess_lowsnr.py output from disk.
        return to_pipeline_image(self.model.apply(self.inner.read_raw(frame_no)))

    def read_raw(self, frame_no):
        return self.model.apply(self.inner.read_raw(frame_no))

    def noise_sd(self):
        """Pixel noise of :meth:`read` frames, in the same (8-bit x 257) units."""
        sd8 = self.model.output_noise_sd(bit_depth=8)
        return None if sd8 is None else sd8 * 257.0

    @property
    def identity(self):
        return self.inner.identity

    def descriptor(self):
        return {"kind": "preprocessed", "inner": self.inner.descriptor(),
                "params": dict(self.params), "model_path": self.model_path}

    @classmethod
    def from_descriptor(cls, d):
        return cls(d["inner"], d["params"], model_path=d.get("model_path"))


def model_filename(params):
    """Per-recipe model file name, so different recipes never share a model."""
    from ...preprocess import params_hash
    return "preprocess_model_%s.npz" % params_hash(params)
