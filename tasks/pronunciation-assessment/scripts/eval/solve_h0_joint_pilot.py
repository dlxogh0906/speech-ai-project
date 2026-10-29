"""Solve only the preregistered lambda=.3 fold problem, with joint hard bounds."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve()
while not (ROOT/'pyproject.toml').exists():
    if ROOT.parent == ROOT:
        raise RuntimeError('repository root not found')
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'tasks/pronunciation-assessment/src'),
               str(ROOT/'tasks/pronunciation-assessment/scripts/train')]
import argparse
import json
import time
import numpy as np
from scipy import sparse
from mdd.h0_joint_weighting import append_joint_constraints, joint_metrics, repair_tiny_residual, repaired_certificate
from mdd.h0_p6_feasibility import certificate, solve_qp, solve_interior
from solve_h0_p6_frontier import load_problem, constraints, objectives, metrics
from train_h0_backbone_control import file_sha
from audit_h0_p5_readiness import write


def problem(root):
    folder = root/'matrices'
    r, z, m = load_problem(folder)
    for name in ('phone_source', 'speaker_source'):
        m[name] = sparse.load_npz(folder/(name+'.npz'))
    A, lo, hi = append_joint_constraints(*constraints(r, z, m), m)
    Q, target = objectives(r, z, m, .3)
    return r, z, m, A, lo, hi, Q, target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--solver', choices=['osqp', 'clarabel'], default='osqp')
    ap.add_argument('--raw-solution',type=Path,help='recheck an existing same-problem iterate without optimizing again')
    a = ap.parse_args(); a.out.mkdir(exist_ok=False)
    protocol = json.loads((a.root/'protocol.json').read_text())
    assert protocol['candidate']['lambda_value'] == .3
    r, z, m, A, lo, hi, Q, target = problem(a.root)
    write(a.out/'protocol.json', dict(created=time.time(), parent=str(a.root/'protocol.json'),
        parent_sha256=file_sha(a.root/'protocol.json'), source_sha256=file_sha(__file__),
        solver=a.solver, lambda_value=.3, model_optimizer_updates=0,
        solver_budget='OSQP up to 40000 iterations; same-problem interior-point fallback if uncertified; no candidate search'))
    def checkpoint(iteration, w, y, cert):
        np.savez(a.out/'iterate.npz', w=w, y=y)
        with (a.out/'convergence.jsonl').open('a') as f:
            f.write(json.dumps(dict(iteration=iteration, **cert))+'\n')
        print('certificate', iteration, json.dumps(cert), flush=True)
    if a.raw_solution:
        raw=np.load(a.raw_solution);w,y=raw['w'],raw['y']
        cert=certificate(Q,target,w,A,lo,hi,y)
        write(a.out/'reuse.json',dict(raw_solution=str(a.raw_solution),sha256=file_sha(a.raw_solution),
            new_numerical_iterations=0,reason='global lower-bound anchor survives feasibility repair'))
    elif a.solver == 'clarabel':
        w, y, cert = solve_interior(Q, target, A, lo, hi)
    else:
        w, y, cert = solve_qp(Q, target, A, lo, hi, np.ones(Q.shape[1]),
                             40000, .1, 0, 1000, checkpoint)
    np.savez(a.out/'raw_solution.npz', w=w, y=y)
    anchor=w.copy()
    repair = None
    if cert['certified_to_tolerance']:
        w, repair = repair_tiny_residual(w, m['source'], A, lo, hi)
    final_cert = repaired_certificate(Q, target, w, anchor, A, lo, hi, y)
    result, groups = metrics(r, z, m, w)
    joint = joint_metrics(m, w)
    result.update(joint_support=joint, raw_certificate=cert, certificate=final_cert,
                  feasibility_repair=repair)
    result['pretraining_treatment_passed'] = bool(result['manipulation_gate_passed'] and
        final_cert['certified_to_tolerance'] and all(x['passed'] for x in joint.values()))
    result['training_allowed'] = False
    result['remaining_before_training'] = protocol['unresolved_before_model_training']
    np.savez(a.out/'solution.npz', w=w, y=y); np.save(a.out/'weights.npy', w)
    expanded = np.ones(r['manifest_rows'])
    expanded[z['ix']] = w[z['row_to_variable']]
    np.save(a.out/'manifest_weights.npy', expanded)
    write(a.out/'report.json', result)
    write(a.out/'finished.json', dict(status='fixed_candidate_numerical_result',
        certified=final_cert['certified_to_tolerance'],
        pretraining_treatment_passed=result['pretraining_treatment_passed'],
        model_optimizer_updates=0, files={p.name:file_sha(p) for p in a.out.iterdir() if p.is_file()}))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
