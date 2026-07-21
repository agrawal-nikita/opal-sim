# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import inspect
import logging
import os

from opal.core.request import LLMRequest
from opal.stats.stage_statistics import StageStatistics
from opal.utils.util import safe_process
from opal.workloads.abstract_workload import AbstractWorkload


def load_class_from_folder(class_name: str):
    # Workload implementations live alongside this module, in the workloads package.
    abs_folder_path = os.path.dirname(__file__)
    this_file = os.path.basename(__file__)
    classes_found = []
    # check the files names
    for filename in os.listdir(abs_folder_path):
        if filename.endswith(".py") and filename not in ("__init__.py", this_file):
            file_path = os.path.join(abs_folder_path, filename)
            module_name = filename[:-3]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # s1 = "straße"
                # s2 = "STRASSE"
                # print(s1.casefold() == s2.casefold())  # True
                # casefold() is stronger than .lower() and handles more Unicode edge cases.
                if name.casefold() == class_name.casefold():
                    return obj
                classes_found.append(name)

    raise ImportError(f"Class {class_name!r} not found in {abs_folder_path}. Found: {classes_found}")


class WorkloadOrchestrator:
    """
    The idea of this class is to generator load-stage-by-stage() to support multi-stage
    load generation with target QPS on different stages. In the case when we are re-playing
    traces, we will not have a target QPS (we can instead calculate QPS from the trace
    while replaying it)

    The idea is quite general and I expect this to expand to tool calling and agentic
    computing workload. With this hook, I can actually generate a DAG of workloads.
    Right now I am just doing a link-list of stages, but the concept is expandable.
    """

    def __init__(self, opal_env: "OpalSimulatorEnvironment"):
        self.opal_env = opal_env
        self.name = "WOrchestrator"
        self.log = logging.getLogger(self.name)
        self.opalConfig = self.opal_env.opalConfig
        self.router = self.opal_env.router
        self.stages: list[AbstractWorkload] = []
        self.stage_stats: list[StageStatistics] = []
        self.workload_orchestration_done = False
        self.active_stage = -1
        # Dynamic loading of the workload stages
        try:
            # this is just a backward compatibility hack
            stages = (
                self.opalConfig["workload"]["stages"]
                if "stages" in self.opalConfig["workload"]
                else [self.opalConfig["workload"]]
            )
            self.log.debug(f"There are {len(stages)} stages of : {stages} workloads")
            for i, s in enumerate(stages):
                self.log.debug(f"Loading {[i]} as {s['type']}")
                workload_class = load_class_from_folder(s["type"])
                # FIXME: remove the dependency of the router from the workload
                self.stages.append(workload_class(self.opal_env, i, s, self.router))
                # allocates the stage-level metrics tracking
                self.stage_stats.append(StageStatistics())
        except:
            str = f"Unknown workload type: {s['type']}."
            raise Exception(str)

    def queue_response_from_router(self, req: LLMRequest):
        stage_id = req.stage_id
        assert stage_id < len(self.stages)
        yield safe_process(self.opal_env.simpy_env, self.stages[stage_id]._queue_response_from_router(req))

    def are_we_done(self) -> bool:
        return self.workload_orchestration_done

    def get_active_stage_stats(self):
        if self.active_stage > -1:
            return self.stage_stats[self.active_stage]
        else:
            None

    def run(self):
        for i, s in enumerate(self.stages):
            self.log.info(f"Stating the execution of stage {i} with type: {str(s)}")
            self.active_stage = i
            self.stage_stats[i].stage_time_start = self.opal_env.simpy_env.now
            # execute each stage in turn and wait until it is done
            yield safe_process(self.opal_env.simpy_env, s._run())
            self.stage_stats[i].stage_time_end = self.opal_env.simpy_env.now
            self.log.info(f"Stage {i} with type: {str(s)} finished")
        self.log.info(f"Workload orchestration finished")
        self.workload_orchestration_done = True

    def get_stage_directory_name(self, stage_id: int):
        """I want to keep the dir name creation in one place"""
        return f"stage_{stage_id}"

    def get_all_stages_statistics(self):
        return self.stage_stats

    def get_num_stages(self):
        return len(self.stage_stats)

    def _process_data(self):
        pass

    def shutdown(self):
        # whatever active stage it was, mark it done
        self.stage_stats[self.active_stage].stage_time_end = self.opal_env.simpy_env.now
        self.log.debug(f"Global shutdown, stage {self.active_stage} aborted.")
        # TODO: write out the stage-by-stage data
