# -*- coding: utf-8 -*-

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse
import scipy as sp
from scipy.special import ndtr   # faster scalar normal CDF than stats.norm.cdf
import colorednoise as cn


# --- Rotation function ---
def rotation(a, theta):
    a = a / np.linalg.norm(a)
    x, y, z = a
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s, x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,   c + y*y*(1-c),   y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,   z*y*(1-c) + x*s, c + z*z*(1-c)]
    ])
    return R


# --- Precomputed parent/child index helpers (no string formatting in hot loop) ---
def parent_index(bIndx):
    """Return the parent branch index using bit shift instead of string formatting."""
    return ((bIndx + 1) >> 1) - 1  # equivalent to int(format(bIndx+1,'b')[:-1], 2) - 1

def child_indices(bIndx, j):
    """Return child branch index (j=0 or 1) using bit shift."""
    return ((bIndx + 1) << 1 | j) - 1  # equivalent to int(format(bIndx+1,'b') + str(j), 2) - 1

def pnode_from_bindx(bIndx):
    """Return pNode (graph node index) for a given branch index."""
    if bIndx == 0:
        return 0
    return (bIndx + 1) >> 1   # int(format(bIndx+1,'b')[:-1], 2)


# --- Vectorized diff eq over all active branches at once ---
def treeDiffEq_batch(pos, angle, vel, thickness, target_pos, angle_norms):
    """
    Compute derivatives for a batch of active branches simultaneously.

    pos:         (N, 3) current positions
    angle:       (N, 3) current angles
    vel:         (N,)   current velocities
    thickness:   (N,)   current thicknesses
    target_pos:  (N, 3) ideal graph node positions (the 'attractor')
    angle_norms: (N,)   precomputed ||angle|| (passed in to avoid recompute)

    Returns dPos, dAngle, dVel, dThickness each shape (N,) or (N,3).
    """
    tVec = pos - target_pos                          # (N, 3)

    tVec_dot = np.einsum('ij,ij->i', tVec, tVec)    # (N,)  ||tVec||^2
    ang_dot_tVec = np.einsum('ij,ij->i', angle, tVec)  # (N,)

    # Project angle onto tVec, then compute perpendicular component
    proj = (ang_dot_tVec / tVec_dot)[:, None] * tVec  # (N, 3)
    dAngle = angle - proj                              # (N, 3)

    # Unit angle * vel
    dPos = (vel / angle_norms)[:, None] * angle       # (N, 3)

    dVel = np.zeros(len(vel))
    dThickness = -thickness**1.3

    return dPos, dAngle, dVel, dThickness


if __name__ == "__main__":
    phi = (1 + np.sqrt(5)) / 2
    parser = argparse.ArgumentParser(description="Get Glider simulation parameters")
    parser.add_argument('--visint', type=int, default=0,
                        help='Display visualization: 1=ideal tree, 2=animated growth')
    parser.add_argument('--tsteps', type=int, default=1000, help='Number of timesteps in growth simulation')
    parser.add_argument('--dt', type=float, default=0.01, help='Temporal step size in growth simulation')
    parser.add_argument('--branchN', type=int, default=5, help='Number of branching iterations for tree outline')
    parser.add_argument('--branchA', type=float, default=2 * np.pi * (1 - 1 / phi), help='Branching angle in tree outline')
    parser.add_argument('--sigStr', type=float, default=10, help='Sigmoid Strength for branching probability')
    parser.add_argument('--nzExp', type=float, default=1, help='Power law exponent used for noise generation')
    parser.add_argument('--nzStr', type=float, default=0.05, help='Strength of noise')
    args = parser.parse_args()
    v = args.visint
    
    thetaStar = args.branchA

    m = args.branchN
    dt = args.dt
    tsteps = args.tsteps
    bStd = args.sigStr

    # Total number of nodes
    a_total = 1 + sum(2**b for b in range(m + 1))

    # Positions
    positions = np.zeros((3, a_total))
    positions[:, 1] = np.array([0, 0, 1])

    # Adjacency matrix (only upper triangle needed)
    AdjMat = np.zeros((a_total, a_total))
    AdjMat[0, 1] = 1

    idl_len = np.concatenate([np.array([1]),np.array([1/(i+1) for i in range(1,m+1) for j in range(1,2**i + 1)])])

    # --- Build tree ---
    for i in range(1, m + 1):
        for j in range(1, 2**i + 1):
            iStar = 2**(i - 1) + (j - 1) // 2
            jStar = 2**i + j - 1

            AdjMat[iStar, jStar] = 1 / (i + 1)
            l = 1 / (i + 1)

            jBin = format(jStar, 'b')

            x = np.array([1, 0, 0])
            y = np.array([0, 1, 0])
            z = np.array([0, 0, 1])

            for bit in jBin:
                if bit == '0':
                    a_vec = rotation(y, np.pi - thetaStar) @ z
                else:
                    a_vec = z

                a2 = rotation(y, np.pi / 2) @ a_vec
                x = rotation(a_vec, thetaStar) @ a2
                y = rotation(a_vec, np.pi / 2) @ x
                z = a_vec

            positions[:, jStar] = positions[:, iStar] + l * a_vec

    # --- Build graph ---
    G = nx.Graph()
    for i in range(a_total):
        G.add_node(i, x=positions[0, i], y=positions[1, i], z=positions[2, i])
        for j in range(i + 1, a_total):
            if AdjMat[i, j] > 0:
                G.add_edge(i, j, weight=AdjMat[i, j])

    # --- Precompute node position array for O(1) lookup (avoid dict hits in loop) ---
    node_pos = positions.T.copy()   # shape (a_total, 3);  node_pos[i] = [x, y, z]
    node_norms = np.linalg.norm(node_pos, axis=1)  # ||node_pos[i]||, precomputed

    # --- Precompute pNode for every branch index ---
    n_branches = a_total - 2**m
    pnode_arr = np.array([pnode_from_bindx(b) for b in range(a_total-1)], dtype=np.int32)

    # --- Plot ideal tree ---
    if v == 1 or v < 0:
        branchOutPlt = np.zeros((a_total-1),dtype=object)
        count = 0
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        print(len(G.edges))
        for (u, vv, data) in G.edges(data=True):
            if v == 1:
                ax.plot([positions[0, u], positions[0, vv]],
                        [positions[1, u], positions[1, vv]],
                        [positions[2, u], positions[2, vv]],
                        linewidth=20 * data['weight'])
            elif v < 0:
                branchOutPlt[count], = ax.plot([positions[0, u], positions[0, vv]],
                        [positions[1, u], positions[1, vv]],
                        [positions[2, u], positions[2, vv]],
                        linewidth=20 * data['weight'])
                count += 1
        if v == 1:
            ax.scatter(positions[0, :], positions[1, :], positions[2, :], s=5)
        elif v < 0:
            nodeOutPlt = ax.scatter(positions[0, :], positions[1, :], positions[2, :], s=5)
        ax.set_axis_off()
        if v == 1:
            plt.show()

    # -----------------------------------------------------------------------
    # Growth simulation
    # -----------------------------------------------------------------------
    #
    # Key changes vs. original:
    #   1. Only store TWO timestep slices (current + next) instead of the full
    #      (n_branches x tsteps x 8) history — cuts RAM and avoids huge indexing.
    #      If you need full history for playback, swap the two-buffer approach
    #      for the original large array and just replace the inner loop logic.
    #   2. Inner loop only iterates over *active* branches (set-based bookkeeping).
    #   3. Batch vectorised diff-eq over all active branches at once.
    #   4. Parent/child indices computed with bit-shifts (no string ops).
    #   5. scipy.special.ndtr replaces stats.norm.cdf (same result, ~10x faster).
    #   6. Node positions read from a pre-built numpy array, not G.nodes dict.
    #
    # -----------------------------------------------------------------------

    # Two rolling buffers: cur_state[b] = state of branch b at time t
    #                      nxt_state[b] = state of branch b at time t+1
    plNoise = np.array([[cn.powerlaw_psd_gaussian(args.nzExp,tsteps-1) for j in range(3)] for i in range(a_total-1)])
    state_shape = (a_total-1, 9)
    cur_state = np.zeros(state_shape)
    nxt_state = np.zeros(state_shape)

    cur_state[0] = [0, 0, 0,  0, 0, 1,  1, 1, 0]   # pos=origin, angle=up, vel=1, thick=1

    # Track which branches are currently growing
    active = set([0])

    if v == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    for t in range(tsteps - 1):

        if not active:
            break  # all done early

        # ---- batch update for all active branches ----
        active_list = np.array(sorted(active), dtype=np.int32)

        pos    = cur_state[active_list, :3]          # (N, 3)
        angle  = cur_state[active_list, 3:6]         # (N, 3)
        vel    = cur_state[active_list, 6]            # (N,)
        thick  = cur_state[active_list, 7]            # (N,)

        # target graph nodes for each active branch (branch bIndx → graph node bIndx+1)
        tgt_idx = active_list + 1                    # graph node indices
        target_pos = node_pos[tgt_idx]               # (N, 3)

        angle_norms = np.linalg.norm(angle, axis=1)  # (N,)

        # Filter out branches with zero angle (shouldn't normally happen, but safe)
        valid = angle_norms > 0
        if not valid.all():
            # copy-forward zero-angle branches without update
            zero_mask = active_list[~valid]
            nxt_state[zero_mask] = cur_state[zero_mask]
            print("No Valid Branches")
            print("Timestep: %d" %t)

        if valid.any():
            vb = active_list[valid]
            dPos, dAngle, dVel, dThick = treeDiffEq_batch(
                pos[valid], angle[valid], vel[valid], thick[valid],
                target_pos[valid], angle_norms[valid]
            )

            nxt_state[vb, :3]  = cur_state[vb, :3]  + dPos  * dt
            nxt_state[vb, 3:6] = cur_state[vb, 3:6] + (dAngle + (args.nzStr * (cur_state[vb,7] * plNoise[vb,:,t].T).T)) * dt
            nxt_state[vb, 6]   = cur_state[vb, 6]   + dVel   * dt
            nxt_state[vb, 7]   = cur_state[vb, 7]   + dThick * dt
            nxt_state[vb, 8]   = cur_state[vb, 8]   + cur_state[vb, 6] * dt

            # ---- check split probability for each valid active branch ----
            pos_nxt   = nxt_state[vb, :3]           # position after step
            pos_cur   = cur_state[vb, :3]

            len_cur = cur_state[vb, 8]
            len_nxt = nxt_state[vb,8]

            # distance from current position to target node
##            diff = pos_cur - target_pos[valid]
##            splitProbNorm = np.linalg.norm(diff, axis=1)   # (N_valid,)
##
##            pos_cur_norm = np.linalg.norm(pos_cur, axis=1)
##            tgt_norm     = node_norms[tgt_idx[valid]]
##
##            sign = np.where(pos_cur_norm > tgt_norm, 1.0, -1.0)

            # idl_len[k] holds the ideal length of graph node k+1, so branch b
            # (target node b+1 = tgt_idx) reads its own length at tgt_idx - 1;
            # indexing by tgt_idx read the child generation's shorter length,
            # making every branch split at roughly half its ideal length
            tgt_len = idl_len[tgt_idx[valid] - 1]
            splitProb = len_cur - tgt_len


            pnodes = pnode_arr[vb]
            scales = bStd*(pnodes + 1)

            # ndtr is scipy.special.ndtr — vectorised standard normal CDF,
            # much faster than calling sp.stats.norm.cdf in a Python loop
##            splitProb = ndtr(sign * splitProbNorm / scales)
            splitProb = 0.5*(np.tanh(scales*splitProb)+1)

            rands = np.random.rand(len(vb))
            split_mask = rands <= splitProb

            to_deactivate = vb[split_mask]
            to_keep       = vb[~split_mask]

            # branches that did NOT split: just carry forward
            # (already written into nxt_state above)

            # branches that DID split: spawn children, deactivate parent
            for idx_in_vb, bIndx in enumerate(to_deactivate):
                active.discard(bIndx)
                spawn_pos = nxt_state[bIndx, :3].copy()
                spawn_vel = nxt_state[bIndx, 6]
                spawn_thk = nxt_state[bIndx, 7]

                for j in range(2):
                    nIndx = child_indices(bIndx, j)
                    if nIndx >= a_total-1:
                        continue   # leaf level — no children exist in array
                    active.add(nIndx)

                    child_node = nIndx + 1
                    direction = node_pos[child_node] - spawn_pos
                    norm_dir  = np.linalg.norm(direction)
                    if norm_dir > 0:
                        direction /= norm_dir

                    nxt_state[nIndx, :3]  = spawn_pos
                    nxt_state[nIndx, 3:6] = direction
                    nxt_state[nIndx, 6]   = spawn_vel
                    nxt_state[nIndx, 7]   = spawn_thk

        # copy-forward inactive branches (needed so history is consistent
        # if you re-enable full history storage)
        inactive = np.array([b for b in range(n_branches) if b not in active], dtype=np.int32)
        if len(inactive):
            nxt_state[inactive] = cur_state[inactive]

        # ---- visualise (mode 2) ----
        if v == 2 or v < 0:
            for bIndx in active:
                x = [cur_state[bIndx, 0], nxt_state[bIndx, 0]]
                y = [cur_state[bIndx, 1], nxt_state[bIndx, 1]]
                z = [cur_state[bIndx, 2], nxt_state[bIndx, 2]]
                ax.plot(x, y, z, linewidth=20 * cur_state[bIndx, 7], color='b')
                ax.set_title("Timestep: %d" %t)
            plt.pause(0.1)

        # ---- swap buffers ----
        cur_state, nxt_state = nxt_state, cur_state

    if v == 2:
        plt.show()
    elif v < 0:
        nodeOutPlt.set_alpha(0)
        for bIndx in range(a_total-1):
            branchOutPlt[bIndx].set_alpha(0)
        plt.show()

    print("Simulation complete.")
