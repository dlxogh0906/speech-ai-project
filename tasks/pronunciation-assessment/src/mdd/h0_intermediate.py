"""Small, identity-initialized heads for the preregistered H0 P3 ablation.

No reference text or TAG timing is an inference input. The signed residual gate
is a simple baseline, not an implementation of self-conditioned/inter-layer CTC.
"""
import torch


class IntermediateHeads(torch.nn.Module):
    def __init__(self, final_head, condition):
        super().__init__()
        if condition not in ('B0', 'B1', 'B2', 'B3'):
            raise ValueError(condition)
        self.condition = condition
        self.norm = torch.nn.LayerNorm(final_head.in_features)
        self.aux = torch.nn.Linear(final_head.in_features, final_head.out_features)
        self.aux.load_state_dict(final_head.state_dict())
        self.gate = torch.nn.Parameter(torch.zeros(()))
        self.use_aux = condition in ('B1', 'B3')
        self.use_fusion = condition in ('B2', 'B3')
        self.norm.requires_grad_(self.use_aux or self.use_fusion)
        self.aux.requires_grad_(self.use_aux)
        self.gate.requires_grad_(self.use_fusion)

    def forward(self, middle, final, auxiliary=False):
        if not (self.use_aux or self.use_fusion):
            return final, None
        middle = self.norm(middle)
        fused = final + self.gate.tanh() * middle if self.use_fusion else final
        aux = self.aux(middle) if auxiliary and self.use_aux else None
        return fused, aux


def rng_state():
    import random
    import numpy as np
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                random=random.getstate(), cuda=torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else [])


def restore_rng(state):
    import random
    import numpy as np
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['random'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])
