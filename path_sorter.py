#!/usr/bin/env python3
"""
Ultra-simple 3D Oriented Turning Handedness Sorter (flat 9-float, pre-normalized + orthonormalized)

v7: Removed preprocessing normalization entirely (fixes double-normalization)
    - Assumes input poses already provide:
        * F (forward, indices 3:6) as unit vector
        * U (up, indices 6:9) as unit vector perpendicular to F
      (i.e. the "pre-normalized + orthonormalized" contract is now fully respected)
    - No Gram-Schmidt, no _normalize3 calls inside fast_coherence_smoothness
    - Single-pass processing (no precompute lists of Fs/Us) → even faster + lower memory
    - Keeps your preferred clean atan2 signed-yaw + real acos bend math
    - All behavior identical when inputs satisfy the pre-normalized+orthonormalized contract
    - Still pure stdlib, ultra-streamlined
"""

import math
from dataclasses import dataclass
from typing import List, Sequence, Optional


@dataclass
class PathScore:
    path: List[Sequence[float]]
    combined: float
    primary: float
    handedness: float
    coherence: float
    smoothness: float


def fast_coherence_smoothness(
    pose_list,
    max_reasonable_bend: float = math.pi / 3.0,
):
    """
    Returns (coherence, smoothness) in [0,1].

    Assumes every pose in pose_list is pre-normalized + orthonormalized:
      - F (forward) at [3], [4], [5] is a unit vector
      - U (up)      at [6], [7], [8] is a unit vector with F · U == 0

    coherence: turning handedness consistency (directional persistence of yaw-ish turns)
               via proper signed yaw = atan2( (F0×F1)·U0 , F0·F1 )
    smoothness: coherence modulated by low average bend angle.
    """
    n = len(pose_list)
    if n < 2:
        return 1.0, 1.0

    total_abs_bend = 0.0
    net_yaw = 0.0
    total_abs_yaw = 0.0

    # First frame (directly from input — already normalized + orthonormal)
    p0 = pose_list[0]
    F0x, F0y, F0z = p0[3], p0[4], p0[5]
    U0x, U0y, U0z = p0[6], p0[7], p0[8]

    for i in range(1, n):
        p1 = pose_list[i]
        F1x, F1y, F1z = p1[3], p1[4], p1[5]
        # We only need the *previous* frame's U (U0) for the signed turn.
        # No need to unpack U from the current pose here.

        dot_f = F0x*F1x + F0y*F1y + F0z*F1z
        c = max(-1.0, min(1.0, dot_f))
        bend = math.acos(c)
        total_abs_bend += bend

        # Cross product F0 × F1
        cx = F0y * F1z - F0z * F1y
        cy = F0z * F1x - F0x * F1z
        cz = F0x * F1y - F0y * F1x
        signed_turn = cx * U0x + cy * U0y + cz * U0z

        # Proper signed yaw component (your preferred clean version)
        signed_yaw = math.atan2(signed_turn, c)

        net_yaw += signed_yaw
        total_abs_yaw += abs(signed_yaw)

        # Advance to next frame (pull next U directly from current p1)
        F0x, F0y, F0z = F1x, F1y, F1z
        U0x, U0y, U0z = p1[6], p1[7], p1[8]

    n_turns = n - 1

    if total_abs_yaw < 1e-9:
        coherence = 1.0  # straight or negligible yaw change → fully consistent
    else:
        coherence = abs(net_yaw) / total_abs_yaw  # 1.0 = perfectly consistent direction

    avg_bend = total_abs_bend / n_turns

    norm_curv = min(1.0, avg_bend / max_reasonable_bend)
    smoothness = coherence * (1.0 - norm_curv)

    return coherence, max(0.0, min(1.0, smoothness))


def rank_paths(
    paths: List[List[Sequence[float]]],
    primary_costs: Optional[List[float]] = None,
    handedness_weight: float = 1.0,
    primary_weight: float = 1.0,
    w_coherence: float = 0.55,
    w_smoothness: float = 0.45,
    lower_is_better_primary: bool = True,
    max_reasonable_bend: float = math.pi / 3.0,
) -> List[PathScore]:
    """Rank paths. Lower combined is better (when lower_is_better_primary=True)."""
    if primary_costs is None:
        primary_costs = [0.0] * len(paths)

    results: List[PathScore] = []
    for path, p_cost in zip(paths, primary_costs):
        coh, sm = fast_coherence_smoothness(
            path,
            max_reasonable_bend=max_reasonable_bend,
        )
        h_score = w_coherence * coh + w_smoothness * sm
        combined = (primary_weight * p_cost - handedness_weight * h_score
                    if lower_is_better_primary else
                    primary_weight * p_cost + handedness_weight * h_score)
        results.append(PathScore(path, combined, p_cost, h_score, coh, sm))

    results.sort(key=lambda x: x.combined)
    return results


def make_straight_path(n: int = 8) -> List[tuple]:
    """All forward same, U fixed. Should give coherence=1.0, smoothness=1.0"""
    path = []
    F = (1.0, 0.0, 0.0)
    U = (0.0, 0.0, 1.0)
    for i in range(n):
        path.append((float(i), 0.0, 0.0, F[0], F[1], F[2], U[0], U[1], U[2]))
    return path


def make_consistent_turn_path(n: int = 8, turn_rate: float = 0.18) -> List[tuple]:
    """Gradual consistent yaw (left or right). High coherence + good smoothness if turn_rate reasonable."""
    path = []
    yaw = 0.0
    for i in range(n):
        F = (math.cos(yaw), math.sin(yaw), 0.0)
        U = (0.0, 0.0, 1.0)
        path.append((float(i), 0.0, 0.0, F[0], F[1], F[2], U[0], U[1], U[2]))
        yaw += turn_rate
    return path


def make_zigzag_path(n: int = 8, turn_rate: float = 0.25) -> List[tuple]:
    """Alternating turn directions → low coherence (cancels in net_yaw)."""
    path = []
    yaw = 0.0
    for i in range(n):
        F = (math.cos(yaw), math.sin(yaw), 0.0)
        U = (0.0, 0.0, 1.0)
        path.append((float(i), 0.0, 0.0, F[0], F[1], F[2], U[0], U[1], U[2]))
        direction = 1.0 if (i % 2 == 0) else -1.0
        yaw += direction * turn_rate
    return path


if __name__ == "__main__":
    print("=== Ultra-simple 3D Oriented Turning Handedness Sorter v7 ===\n")

    path_straight = make_straight_path(8)
    path_left     = make_consistent_turn_path(8, turn_rate=0.15)
    path_zigzag   = make_zigzag_path(8, turn_rate=0.22)

    paths = [path_straight, path_left, path_zigzag]
    primary_costs = [4.0, 4.1, 3.9]

    print("Ranking (lower combined better):")
    ranked = rank_paths(paths, primary_costs=primary_costs)
    for i, ps in enumerate(ranked, 1):
        print(f"  {i}. coh={ps.coherence:.4f} | smooth={ps.smoothness:.4f} | "
              f"h_score={ps.handedness:.4f} | combined={ps.combined:.4f}")

    print("\nExpected behavior:")
    print("  - Straight & consistent-turn paths → coh ≈ 1.000, high smoothness")
    print("  - Zigzag path → coh << 1.0 (direction reversals cancel net_yaw)")