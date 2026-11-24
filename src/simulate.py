import numpy as np
import pandas as pd

from src.models import RLParams, softmax_choice_prob, update_q_values


def generate_session_contexts(
    rng: np.random.Generator,
    n_trials_per_context: int = 24,
) -> list[dict]:
    """Generate the 4 contexts (24 trials each = 96 trials) for one session."""
    half = n_trials_per_context // 2
    contexts = []

    p_sym = np.full((n_trials_per_context, 2), 0.5)
    contexts.append({
        "condition": "symmetric",
        "context_idx": 0,
        "p_reward": p_sym,
        "best_action_first_half": None,
        "best_action_second_half": None,
    })

    for asym_idx in range(2):
        best_opt = int(rng.integers(0, 2))
        p_asym = np.full((n_trials_per_context, 2), 0.25)
        p_asym[:, best_opt] = 0.75
        contexts.append({
            "condition": "asymmetric",
            "context_idx": 1 + asym_idx,
            "p_reward": p_asym,
            "best_action_first_half": best_opt,
            "best_action_second_half": best_opt,
        })

    init_best = int(rng.integers(0, 2))
    other_opt = 1 - init_best
    p_rev = np.zeros((n_trials_per_context, 2))
    p_rev[:half, init_best] = 0.83
    p_rev[:half, other_opt] = 0.17
    p_rev[half:, init_best] = 0.17
    p_rev[half:, other_opt] = 0.83
    contexts.append({
        "condition": "reversal",
        "context_idx": 3,
        "p_reward": p_rev,
        "best_action_first_half": init_best,
        "best_action_second_half": other_opt,
    })

    rng.shuffle(contexts)
    for order_idx, ctx in enumerate(contexts):
        ctx["context_order"] = order_idx
    return contexts


def simulate_participant(
    participant_id: int,
    params: RLParams,
    experiment: int = 2,
    n_sessions: int = 2,
    n_trials_per_context: int = 24,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate one virtual participant across 2 sessions x 4 contexts x 24 trials (192 trials)."""
    if experiment not in (1, 2):
        raise ValueError("experiment must be 1 (partial feedback) or 2 (complete feedback).")

    rng = np.random.default_rng(seed)
    counterfactual = experiment == 2
    half = n_trials_per_context // 2
    rows = []
    global_trial = 0

    for session in range(1, n_sessions + 1):
        contexts = generate_session_contexts(rng, n_trials_per_context=n_trials_per_context)
        for ctx in contexts:
            q_values = np.zeros(2, dtype=float)
            prev_choice: int | None = None
            p_reward = ctx["p_reward"]

            for t in range(n_trials_per_context):
                global_trial += 1
                probs = softmax_choice_prob(
                    q_values=q_values,
                    beta=params.beta,
                    prev_choice=prev_choice,
                    pi=params.perseveration,
                )
                choice = int(rng.choice(2, p=probs))
                unchosen = 1 - choice

                r0 = 1.0 if rng.random() < p_reward[t, 0] else -1.0
                r1 = 1.0 if rng.random() < p_reward[t, 1] else -1.0
                outcomes = np.array([r0, r1], dtype=float)

                r_chosen = float(outcomes[choice])
                r_unchosen = float(outcomes[unchosen]) if counterfactual else np.nan

                phase = "first_half" if t < half else "second_half"
                if ctx["condition"] == "asymmetric":
                    is_correct = int(choice == ctx["best_action_first_half"])
                elif ctx["condition"] == "reversal":
                    target = ctx["best_action_first_half"] if t < half else ctx["best_action_second_half"]
                    is_correct = int(choice == target)
                else:
                    is_correct = np.nan

                q_next, pe_info = update_q_values(
                    q_values=q_values,
                    choice=choice,
                    r_chosen=r_chosen,
                    r_unchosen=None if not counterfactual else r_unchosen,
                    params=params,
                    counterfactual=counterfactual,
                )

                rows.append({
                    "participant_id": participant_id,
                    "experiment": experiment,
                    "session": session,
                    "context_order": ctx["context_order"],
                    "condition": ctx["condition"],
                    "phase": phase,
                    "trial_in_context": t + 1,
                    "global_trial": global_trial,
                    "choice": choice,
                    "unchosen": unchosen,
                    "r_chosen": r_chosen,
                    "r_unchosen": r_unchosen,
                    "p_choice_0": probs[0],
                    "p_choice_1": probs[1],
                    "q_0_before": q_values[0],
                    "q_1_before": q_values[1],
                    "is_correct": is_correct,
                    "pe_c": pe_info["pe_c"],
                    "pe_u": pe_info.get("pe_u", np.nan),
                })

                q_values = q_next
                prev_choice = choice

    return pd.DataFrame(rows)


def simulate_cohort(
    param_list: list[RLParams],
    experiment: int = 2,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Simulate a cohort of participants given a list of RLParams."""
    dfs = []
    for idx, p in enumerate(param_list):
        df_sub = simulate_participant(
            participant_id=idx + 1,
            params=p,
            experiment=experiment,
            seed=base_seed + idx,
        )
        dfs.append(df_sub)
    return pd.concat(dfs, ignore_index=True)
