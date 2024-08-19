from typing import List


class BaseDict:
    def __init__(self, val: dict):
        self.val = val

    def dict_to_list(self) -> List[str]:
        return [f"{key}:{value}" for key, value in self.val.items()]


class SourceDict(BaseDict):
    def __repr__(self):
        return f"SourceDict(source_list={self.val})"


class TargetDict(BaseDict):
    def __repr__(self):
        return f"TargetDict(target_list={self.val})"
