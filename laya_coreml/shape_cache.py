"""Bounded Core ML execution instances for alternating input shapes."""

import threading
from collections import OrderedDict


class ShapeCache:
    def __init__(self, capacity=4):
        if type(capacity) is not int or not 0 <= capacity <= 32:
            raise ValueError("shape_cache_size must be an integer in 0..32")
        self.capacity = capacity
        self._models = OrderedDict()
        self._frequency = {}
        self._misses = OrderedDict()
        self._base_shapes = {}
        self._requests = 0
        self._lock = threading.RLock()

    def _remember(self, key, frequency):
        self._misses[key] = max(frequency, self._misses.get(key, 0))
        self._misses.move_to_end(key)
        while len(self._misses) > max(16, self.capacity * 4):
            self._misses.popitem(last=False)

    def predict(self, model, batch):
        if not self.capacity:
            return model.predict(batch)
        # All input shapes belong to the key: parallel layouts and K also matter.
        key = (model.package_path, tuple(sorted((k, v.shape) for k, v in batch.items())))
        # Keep selection, eviction and prediction together. An evicted instance
        # cannot remain in flight and exceed the configured resident capacity.
        with self._lock:
            self._requests += 1
            if self._requests % 64 == 0:
                # Age popularity so an old hot workload does not own slots forever.
                self._frequency = {k: max(1, n // 2) for k, n in self._frequency.items()}
                self._misses = OrderedDict((k, max(1, n // 2)) for k, n in self._misses.items())
                self._base_shapes = {
                    path: (shape, max(1, n // 2)) for path, (shape, n) in self._base_shapes.items()
                }
            base = self._base_shapes.get(model.package_path)
            if key not in self._models and (base is None or base[0] == key):
                # The already loaded physical graph supplies its first/current
                # shape for free. Reserve extra instances for additional shapes.
                self._base_shapes[model.package_path] = (key, 1 if base is None else base[1] + 1)
                return model.predict(batch)
            if key not in self._models:
                frequency = self._misses.get(key, 0) + 1
                self._remember(key, frequency)
                if len(self._models) >= self.capacity:
                    victim = min(self._models, key=self._frequency.__getitem__)
                    if frequency <= self._frequency[victim]:
                        # Long-tail shapes use the existing graph. Loading a new
                        # instance on every miss is much slower than reshaping it.
                        self._remember(*base)
                        self._misses.pop(key, None)
                        self._base_shapes[model.package_path] = (key, frequency)
                        return model.predict(batch)
                    del self._models[victim]
                    del self._frequency[victim]
                import coremltools as ct

                # The owning Agent retains the base model and therefore its
                # compiled directory for the lifetime of these instances.
                # Recompiling the package on each admission can crash inside
                # coremltools' native proxy/GIL handling on macOS 26.
                self._models[key] = ct.models.CompiledMLModel(
                    model.get_compiled_model_path(), compute_units=model.compute_unit
                )
                self._frequency[key] = frequency
                self._misses.pop(key, None)
            else:
                self._frequency[key] += 1
            self._models.move_to_end(key)
            return self._models[key].predict(batch)

    def clear(self):
        with self._lock:
            self._models.clear()
            self._frequency.clear()
            self._misses.clear()
            self._base_shapes.clear()
            self._requests = 0
