"""Convex TRAIN contribution diagnostics, with an independently evaluable bound.

This module does not train speech models. The box-supported first-order dual
bound remains valid for arbitrary constraint multipliers, including an
unfinished numerical solve. All arithmetic is floating point, not interval proof.
"""
import numpy as np
from scipy import sparse


def quadratic_value_gradient(Q, target, w):
    residual = Q.dot(w) - target
    return float(residual.dot(residual)), 2 * Q.T.dot(residual)


def certificate(Q, target, w, A, lower, upper, multipliers, lo=.25, hi=4.):
    """Lower bound from convex tangent + linear constraints + exact box support.

    For l <= A x <= u and any y, y A x <= y+ u + y- l.
    Minimize the tangent plus y A x minus that support over the box.
    The result is a lower bound even if w itself is infeasible.
    """
    value, gradient = quadratic_value_gradient(Q, target, w)
    y = np.asarray(multipliers).copy()
    # A multiplier with the wrong sign on a one-sided constraint is invalid.
    y[(~np.isfinite(upper)) & (y > 0)] = 0
    y[(~np.isfinite(lower)) & (y < 0)] = 0
    pos, neg = y > 0, y < 0
    support = float(y[pos].dot(upper[pos]) + y[neg].dot(lower[neg]))
    g = gradient + A.T.dot(y)
    box_min = float(np.minimum(lo * g, hi * g).sum())
    lb = value - float(gradient.dot(w)) + box_min - support
    aw = A.dot(w)
    violation = max(float(np.maximum(lower - aw, 0).max()),
                    float(np.maximum(aw - upper, 0).max()),
                    float(np.maximum(lo - w, 0).max()),
                    float(np.maximum(w - hi, 0).max()))
    return dict(objective=value, lower_bound=lb, upper_minus_lower=value-lb,
                max_constraint_violation=violation,
                numerically_feasible=violation <= 2e-6,
                certified_to_tolerance=(violation <= 2e-6 and
                    abs(value-lb) <= 2e-5 * max(1., abs(value))),
                bound_scope='convex tangent bound in float64; not interval arithmetic')


def solve_qp(Q, target, A, lower, upper, initial=None, max_iter=100000,
             rho=.1, adaptive_interval=0, checkpoint_every=0, checkpoint=None):
    """Sparse residual-variable QP; never materializes Q.T @ Q."""
    import osqp
    n, k = Q.shape[1], Q.shape[0]
    # Minimize ||z||^2 subject to Qw-z=target and the original constraints.
    P = sparse.diags(np.r_[np.zeros(n), np.full(k, 2.)], format='csc')
    C = sparse.vstack([
        sparse.hstack([Q, -sparse.eye(k)]),
        sparse.hstack([A, sparse.csr_matrix((A.shape[0], k))]),
        sparse.hstack([sparse.eye(n), sparse.csr_matrix((n, k))])], format='csc')
    l = np.r_[target, lower, np.full(n, .25)]
    u = np.r_[target, upper, np.full(n, 4.)]
    solver = osqp.OSQP()
    solver.setup(P=P, q=np.zeros(n+k), A=C, l=l, u=u, eps_abs=1e-8,
                 eps_rel=1e-8, max_iter=max_iter, polish=True,
                 verbose=True, adaptive_rho=True, check_termination=100,
                 rho=rho, adaptive_rho_interval=adaptive_interval)
    if initial is not None:
        solver.warm_start(x=np.r_[initial, Q.dot(initial)-target])
    total=0;runtime=0.
    while total<max_iter:
        if checkpoint_every:solver.update_settings(max_iter=min(checkpoint_every,max_iter-total))
        result = solver.solve()
        if result.x is None or not np.all(np.isfinite(result.x)):
            raise RuntimeError('No finite solver iterate: '+result.info.status)
        w = result.x[:n]
        y = result.y[k:k+A.shape[0]]
        c = certificate(Q, target, w, A, lower, upper, y)
        total+=result.info.iter;runtime+=result.info.run_time
        if checkpoint is not None:checkpoint(total,w,y,c)
        if c['certified_to_tolerance'] or not checkpoint_every or result.info.status=='solved':break
    c.update(solver_status=result.info.status, iterations=total,
             primal_residual=result.info.pri_res, dual_residual=result.info.dua_res,
             solver_run_time=runtime, osqp_version=osqp.__version__)
    return w, y, c


def solve_interior(Q,target,A,lower,upper):
    """Independent interior-point implementation of the same residual QP."""
    import clarabel
    n,k=Q.shape[1],Q.shape[0]
    eq=np.flatnonzero(np.isfinite(lower)&np.isfinite(upper)&(lower==upper))
    up=np.flatnonzero(np.isfinite(upper)&(lower!=upper))
    low=np.flatnonzero(np.isfinite(lower)&(lower!=upper))
    pad=lambda M:sparse.hstack([M,sparse.csr_matrix((M.shape[0],k))])
    C=sparse.vstack([sparse.hstack([Q,-sparse.eye(k)]),pad(A[eq]),pad(A[up]),pad(-A[low]),pad(sparse.eye(n)),pad(-sparse.eye(n))],format='csc')
    b=np.r_[target,upper[eq],upper[up],-lower[low],np.full(n,4.),np.full(n,-.25)]
    P=sparse.diags(np.r_[np.zeros(n),np.full(k,2.)],format='csc')
    cones=[clarabel.ZeroConeT(k+len(eq)),clarabel.NonnegativeConeT(len(up)+len(low)+2*n)]
    settings=clarabel.DefaultSettings();settings.max_iter=100
    settings.tol_gap_abs=1e-8;settings.tol_gap_rel=1e-8;settings.tol_feas=1e-9
    solver=clarabel.DefaultSolver(P,np.zeros(n+k),C,b,cones,settings)
    result=solver.solve();w=np.asarray(result.x[:n]);dual=np.asarray(result.z);y=np.zeros(A.shape[0]);start=k
    y[eq]=dual[start:start+len(eq)];start+=len(eq)
    y[up]+=dual[start:start+len(up)];start+=len(up)
    y[low]-=dual[start:start+len(low)]
    c=certificate(Q,target,w,A,lower,upper,y)
    c.update(solver_status=str(result.status),iterations=result.iterations,
        primal_residual=result.r_prim,dual_residual=result.r_dual,solver_run_time=result.solve_time,
        osqp_version=None,clarabel_version=clarabel.__version__)
    return w,y,c
