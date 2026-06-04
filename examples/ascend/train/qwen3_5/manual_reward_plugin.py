from typing import List

import torch.distributed as dist

from swift.rewards import ORM, orms


class ManualAlternatingReward(ORM):
    """Smoke-only reward that keeps GRPO reward variance non-zero."""

    def __call__(self, completions, **kwargs) -> List[float]:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        base = rank * max(len(completions), 1)
        return [float((base + i) % 2) for i, _ in enumerate(completions)]


orms['manual_alternating'] = ManualAlternatingReward
