from enum import Enum as E


class ExtendedEnum(E):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
