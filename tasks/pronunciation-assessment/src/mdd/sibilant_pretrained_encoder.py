"""Load the original encoder without constructing a nonexistent CTC head."""
import torch


def load_encoder(protocol, tuned):
    from transformers import Wav2Vec2BertModel
    encoder, info = Wav2Vec2BertModel.from_pretrained(protocol["checkpoint"], output_loading_info=True)
    if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("pretrained encoder did not load exactly: " + repr(info))
    print("pretrained encoder loaded: no missing, unexpected or mismatched weights", flush=True)
    encoder.requires_grad_(False)
    if tuned:
        for layer in encoder.encoder.layers[-2:]:
            layer.requires_grad_(True)
    encoder.eval()
    device = protocol.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    return encoder.to(torch.device(device))
