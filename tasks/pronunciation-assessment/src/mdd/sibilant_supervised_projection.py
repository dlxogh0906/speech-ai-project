"""Learn the fixed-size projection without changing scaler or audio encoder."""
import torch
from mdd.sibilant_frozen_probe import FrozenProbe, READOUT, projection_digest
from mdd.sibilant_vowel_presence import CLASSES


class SupervisedProjectionProbe(FrozenProbe):
    def __init__(self, projection):
        super().__init__(projection)
        self.initial_variance = projection['explained_variance_ratio']
        self.components = torch.nn.Parameter(self.components)

    def export_projection(self):
        # A learned matrix need not remain orthogonal. The PCA variance ratio
        # describes the initial matrix only and must not label the learned one.
        result = {key: getattr(self, key).detach().cpu().clone()
                  for key in ('mean', 'scale', 'components')}
        result.update(training='supervised_matrix_v1',
                      initial_pca_explained_variance_ratio=self.initial_variance)
        return result


def supervised_artifact(model, binding):
    projection = model.export_projection()
    return {'readout': READOUT,
            'binding': dict(binding, classes=list(CLASSES),
                            projection_sha256=projection_digest(projection),
                            projection_training='supervised_matrix_v1'),
            'projection': projection,
            'head': {key: value.detach().cpu().clone()
                     for key, value in model.head.state_dict().items()}}
