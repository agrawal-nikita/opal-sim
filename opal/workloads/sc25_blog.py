# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import json
import logging
from random import Random
from opal.core.request import LLMRequest
from opal.utils.util import safe_process
from opal.workloads.abstract_workload import AbstractWorkload
import numpy as np


class SC25Workload(AbstractWorkload):
    def __init__(
        self,
        opal_env: "OpalSimulatorEnvironment",
        stage_id: int,
        workload_params: dict,
        req_router,
        name: str = "SC25Workload",
    ):
        super().__init__(opal_env, stage_id, workload_params, req_router, name, False)
        self.is_finished = False
        self.rng: Random = self.opal_env.get_fresh_random_variable()
        self.log = logging.getLogger(self.name)
        self._run()

    def generate_requests(self):
        """This is a very restrictive workload with specific prompt sizes to generate."""

        json_data = """
        {
            "gpu": {
                "4092": "312.65",
                "8184": "618.24",
                "16368": "1314.7",
                "32736": "2905.65",
                "49104": "4827.76",
                "65472": "7090.23",
                "81840": "9579.79",
                "98208" : "12465.68",
                "114576": "15468.25",
                "130944": "18899.58"
            }, 
            "params": {
                "divider_to_seconds": "1000" 
            }
            }
        """
        data1 = json.loads(json_data)
        prompt_sizes = data1["gpu"].keys()
        self.request_id = 0
        # cold run first
        content = []
        for p in prompt_sizes:
            intp = int(p)
            # make the class 'numpy.ndarray' to python tolist()
            # otherwise it will not match the list[int] matching types
            hash_ids = [self.rng.randrange(0, 10**10) for _ in range(intp)]
            request = LLMRequest(self.simpy_env, self.stage_id, int(p), hash_ids=hash_ids)
            content.append(hash_ids)
            self.request_id += 1
            # UGLY
            self.generated_requests += 1
            self.log.debug(f"cold {request} generated")
            yield self.req_router.input_queue.put(request)
            # wait until it is completed
            # NOTE: get_completed_requests() is a generator, so we must use 'yield from'
            # Using 'yield' alone would try to yield the generator object itself (invalid)
            # Using 'yield from' properly delegates to the generator and yields its values
            completed = yield safe_process(self.simpy_env, self.get_completed_requests())

        self.log.debug(f"Cold workload generation finished with {self.generated_requests} (cumulative) requests ")
        self.log.info(f"Starting the hot round with 100% cache hit")

        for h, p in zip(content, prompt_sizes):
            intp = int(p)
            request = LLMRequest(self.simpy_env, self.stage_id, int(p), hash_ids=h)
            self.request_id += 1
            # UGLY
            self.generated_requests += 1
            self.log.debug(f"warm targeted 100% hit {request} generated")
            yield self.req_router.input_queue.put(request)
            # wait until it is completed
            # NOTE: get_completed_requests() is a generator, so we must use 'yield from'
            # Using 'yield' alone would try to yield the generator object itself (invalid)
            # Using 'yield from' properly delegates to the generator and yields its values
            completed = yield safe_process(self.simpy_env, self.get_completed_requests())

        self.log.info(f"Warm workload generation finished with {self.generated_requests} (cumulative) requests ")
        self.is_finished = True
