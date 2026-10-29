"""Typed local-phone CTC tails; preserve the original two-layer artifact format."""
import torch

SCHEMA = 'runtime_nextV_original20left11right_presence_v1'


def validate_depth(depth, total):
    if type(depth) is not int or not 1 <= depth <= total:
        raise ValueError('encoder tail depth must be an integer within the model')
    return depth


def make_artifact(tail, head, depth, vocab_sha256, objective):
    if len(tail) != depth:
        raise ValueError('tail length differs from declared depth')
    weights = {k: v.detach().cpu().clone() for k, v in tail.state_dict().items()}
    artifact = {'readout': 'local_phone_ctc_last2_v1' if depth == 2 else 'local_phone_ctc_tail_v2',
                'coordinate_schema': SCHEMA, 'vocab_sha256': vocab_sha256,
                'lm_head': {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                'training_objective': objective}
    if depth == 2:
        artifact['last2'] = weights
    else:
        artifact.update(tail=weights, encoder_train_layers=depth)
    return artifact


def _validate_weights(module, weights):
    expected = module.state_dict()
    if set(expected) != set(weights):
        raise ValueError('local CTC weight names differ from declared tail/head')
    if any(not isinstance(weights[k], torch.Tensor) or weights[k].shape != v.shape
           for k, v in expected.items()):
        raise ValueError('local CTC weight shapes differ from model')


def load_artifact(layers, head, artifact, vocab_sha256):
    if artifact.get('coordinate_schema') != SCHEMA or artifact.get('vocab_sha256') != vocab_sha256:
        raise ValueError('matching local CTC coordinates and vocabulary required')
    if artifact.get('readout') == 'local_phone_ctc_last2_v1':
        depth, key = 2, 'last2'
        if artifact.get('encoder_train_layers', 2) != 2:
            raise ValueError('legacy local CTC artifact must contain two layers')
    elif artifact.get('readout') == 'local_phone_ctc_tail_v2':
        depth, key = artifact.get('encoder_train_layers'), 'tail'
    else:
        raise ValueError('unsupported local CTC artifact type')
    validate_depth(depth, len(layers))
    tail = layers[-depth:]
    # Validate both components before mutating either; a bad artifact must not
    # leave a half-loaded decoder in the caller.
    _validate_weights(tail, artifact.get(key, {}))
    _validate_weights(head, artifact.get('lm_head', {}))
    tail.load_state_dict(artifact[key])
    head.load_state_dict(artifact['lm_head'])
    return depth
