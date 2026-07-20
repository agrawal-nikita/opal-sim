# SPDX-License-Identifier: Apache-2.0
class AutoScaling:
    """The logic in this class deals with auto-scaling logic"""

    def __init__(self, env, **kwargs):
        self.workers = 10
