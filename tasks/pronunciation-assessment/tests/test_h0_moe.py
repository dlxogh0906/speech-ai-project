import torch
from mdd.h0_moe import ResidualExperts, routing_counts


def test_identity_and_equal_initialization():
    torch.manual_seed(7)
    a = ResidualExperts(8, 3, 4, True)
    torch.manual_seed(7)
    b = ResidualExperts(8, 3, 4, False)
    z, h = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 2])
    for model in (a, b):
        state = torch.get_rng_state()
        out, loss, gates = model(z, h, lengths)
        assert torch.equal(out, h) and loss.item() == 0
        assert torch.equal(state, torch.get_rng_state())
        assert torch.all(gates == .25)
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other and torch.equal(p, q)


def test_padding_excluded_from_routing_balance():
    torch.manual_seed(8)
    m = ResidualExperts(8, 3, 4, True)
    torch.nn.init.normal_(m.router.weight)
    z, h = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    lengths = torch.tensor([5, 2])
    _, before, gates = m(z, h, lengths)
    z[1, 2:] = torch.randn(3, 8) * 100
    h[1, 2:] = torch.randn(3, 8) * 100
    _, after, other = m(z, h, lengths)
    assert torch.equal(before, after)
    assert routing_counts(gates, lengths) == routing_counts(other, lengths)
    assert routing_counts(gates, lengths)['frames'] == 7


def test_router_and_all_experts_receive_task_gradients():
    torch.manual_seed(9)
    m = ResidualExperts(8, 3, 4, True)
    opt = torch.optim.AdamW(m.parameters(), lr=.01)
    z, h, target = [torch.randn(2, 5, 8) for _ in range(3)]
    lengths = torch.tensor([5, 3])
    for _ in range(3):
        opt.zero_grad()
        out, _, gates = m(z, h, lengths)
        loss = (out-target).square().mean()
        loss.backward()
        opt.step()
    assert m.router.weight.grad.abs().sum() > 0
    for expert in m.experts:
        assert expert[0].weight.grad.abs().sum() > 0
        assert expert[-1].weight.grad.abs().sum() > 0
    out, _, gates = m(z, h, lengths)
    uniform, _, _ = m(z, h, lengths, force_uniform=True)
    assert not torch.equal(out, uniform)
    assert gates.var(dim=1).sum() > 0
    assert torch.allclose(gates.sum(-1), torch.ones(2, 5))


def test_routing_ignores_padded_nonblank_predictions():
    g = torch.full((2, 4, 4), .25)
    pred = torch.tensor([[0, 4, 0, 5], [4, 5, 5, 5]])
    d = routing_counts(g, torch.tensor([4, 1]), pred)
    assert d['frames'] == 3 and d['mass'] == [.75] * 4


def test_large_uniform_frame_reduction_is_zero():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = ResidualExperts(8, 3, 4, True).to(device)
    z = torch.randn(17, 713, 8, device=device)
    lengths = torch.arange(500, 517, device=device)
    _, loss, _ = m(z, z, lengths)
    assert abs(loss.item()) <= 1e-12
    loss.backward()
    assert m.router.weight.grad.abs().max().item() <= 1e-12
