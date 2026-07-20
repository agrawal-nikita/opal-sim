# SPDX-License-Identifier: Apache-2.0
class NoDynamicAttributesMeta(type):
    def __new__(cls, name, bases, dct):
        # Collect all annotations for this class
        annotations = dct.get("__annotations__", {})
        # Use annotations as slots
        dct["__slots__"] = tuple(annotations.keys())
        return super().__new__(cls, name, bases, dct)


class NoDynamicAttributes(metaclass=NoDynamicAttributesMeta):
    pass
