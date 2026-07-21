# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from opal.router.router import Router
from opal.worker.vllm_worker import LLMWorkerVLLMScheduler

"""
This object resolves the issue of looking up different objects based on their indexes. 
It helps to avoid useless circular dependencies. 

For example, a workers needs to instantiate a KVCache object, while internally KVCache managed need to know how to generate 
  KVCEVents and send to the worker -> Router. 

This can also be used to lookup from id -> worker, so that we do not have to pass around internal router "queues" to a worker 

It is up to the implementation to keep these registries up-to-date 
"""


class OpalRegistry:
    _instance = None

    # singleton class init
    def __init__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # when this object is passed here after () call from the above code
        if not hasattr(self, "init_done"):
            self.init_done = True
            self._registry_worker: dict[int, LLMWorkerVLLMScheduler] = {}
            self._router: Router = None

    def add_worker(self, w: LLMWorkerVLLMScheduler):
        if not (w.id in self._registry_worker):
            self._registry_worker[w.id] = w

    def get_worker(self, id: int):
        assert id in self._registry_worker
        return self._registry_worker[id]

    def del_worker(self, id: int):
        self._registry_worker.pop(id, None)

    def put_router(self, r: Router):
        self._router = r

    def get_router(self):
        assert self._router is not None
        return self._router
