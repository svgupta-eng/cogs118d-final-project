import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.models import RLParams, softmax_choice_prob, update_q_values


MODEL_PARAM_COUNTS = {
    "one": 2,           # beta, alpha
    "valence": 3,       # beta, alpha_pos, alpha_neg
    "information": 3,   # beta, alpha_c, alpha_u (in exp 2; 2 in exp 1)
    "confirmation": 3,  # beta, alpha_con, alpha_dis
    "full": 5,          # beta, alpha_c_pos, alpha_c_neg, alpha_u_pos, alpha_u_neg (3 in exp 1)
    "perseveration": 4, # beta, alpha_con, alpha_dis, pi
}


def vector_to_params(theta: np.ndarray, model_name: str, counterfactual: bool = True) -> RLParams:
    """Map an unconstrained/bounded optimizer vector theta to RLParams."""
    if model_name == "one":
        beta, alpha = float(theta[0]), float(theta[1])
        return RLParams(
            beta=beta,
            alpha_c_pos=alpha,
            alpha_c_neg=alpha,
            alpha_u_pos=alpha,
            alpha_u_neg=alpha,
            perseveration=0.0,
        )
    if model_name == "valence":
        beta, a_pos, a_neg = float(theta[0]), float(theta[1]), float(theta[2])
        return RLParams(
            beta=beta,
            alpha_c_pos=a_pos,
            alpha_c_neg=a_neg,
            alpha_u_pos=a_pos,
            alpha_u_neg=a_neg,
            perseveration=0.0,
        )
    if model_name == "information":
        beta, a_c = float(theta[0]), float(theta[1])
        a_u = float(theta[2]) if counterfactual and len(theta) > 2 else a_c
        return RLParams(
            beta=beta,
            alpha_c_pos=a_c,
            alpha_c_neg=a_c,
            alpha_u_pos=a_u,
            alpha_u_neg=a_u,
            perseveration=0.0,
        )
    if model_name == "confirmation":
        beta, a_con, a_dis = float(theta[0]), float(theta[1]), float(theta[2])
        return RLParams.from_confirmation(
            beta=beta,
            alpha_con=a_con,
            alpha_dis=a_dis,
            perseveration=0.0,
        )
    if model_name == "full":
        beta, a_cp, a_cn = float(theta[0]), float(theta[1]), float(theta[2])
        if counterfactual and len(theta) >= 5:
            a_up, a_un = float(theta[3]), float(theta[4])
        else:
            a_up, a_un = a_cp, a_cn
        return RLParams(
            beta=beta,
            alpha_c_pos=a_cp,
            alpha_c_neg=a_cn,
            alpha_u_pos=a_up,
            alpha_u_neg=a_un,
            perseveration=0.0,
        )
    if model_name == "perseveration":
        beta, a_con, a_dis, pi = float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])
        return RLParams.from_confirmation(
            beta=beta,
            alpha_con=a_con,
            alpha_dis=a_dis,
            perseveration=pi,
        )
    raise ValueError(f"Unknown model_name: {model_name}")


def get_bounds_and_init(model_name: str, counterfactual: bool = True) -> tuple[list[tuple[float, float]], np.ndarray]:
    """Return parameter bounds and default starting point for each model."""
    eps = 1e-4
    if model_name == "one":
        bounds = [(0.05, 15.0), (eps, 1.0 - eps)]
        x0 = np.array([3.0, 0.35])
    elif model_name == "valence":
        bounds = [(0.05, 15.0), (eps, 1.0 - eps), (eps, 1.0 - eps)]
        x0 = np.array([3.0, 0.45, 0.25])
    elif model_name == "information":
        if counterfactual:
            bounds = [(0.05, 15.0), (eps, 1.0 - eps), (eps, 1.0 - eps)]
            x0 = np.array([3.0, 0.35, 0.35])
        else:
            bounds = [(0.05, 15.0), (eps, 1.0 - eps)]
            x0 = np.array([3.0, 0.35])
    elif model_name == "confirmation":
        bounds = [(0.05, 15.0), (eps, 1.0 - eps), (eps, 1.0 - eps)]
        x0 = np.array([3.0, 0.50, 0.20])
    elif model_name == "full":
        if counterfactual:
            bounds = [(0.05, 15.0)] + [(eps, 1.0 - eps)] * 4
            x0 = np.array([3.0, 0.50, 0.20, 0.20, 0.50])
        else:
            bounds = [(0.05, 15.0)] + [(eps, 1.0 - eps)] * 2
            x0 = np.array([3.0, 0.50, 0.20])
    elif model_name == "perseveration":
        bounds = [(0.05, 15.0), (eps, 1.0 - eps), (eps, 1.0 - eps), (-5.0, 5.0)]
        x0 = np.array([3.0, 0.50, 0.20, 0.0])
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    return bounds, x0


def compute_nll(df_sub: pd.DataFrame, params: RLParams, counterfactual: bool = True) -> float:
    """Compute negative log-likelihood of observed choices for one participant."""
    total_ll = 0.0
    for (_, _), ctx_df in df_sub.groupby(["session", "context_order"], sort=False):
        ctx_df = ctx_df.sort_values("trial_in_context")
        q_values = np.zeros(2, dtype=float)
        prev_choice: int | None = None

        for row in ctx_df.itertuples(index=False):
            probs = softmax_choice_prob(
                q_values=q_values,
                beta=params.beta,
                prev_choice=prev_choice,
                pi=params.perseveration,
            )
            choice = int(row.choice)
            p_chosen = float(np.clip(probs[choice], 1e-12, 1.0))
            total_ll += np.log(p_chosen)

            r_chosen = float(row.r_chosen)
            r_unchosen = None if (not counterfactual or pd.isna(row.r_unchosen)) else float(row.r_unchosen)
            q_values, _ = update_q_values(
                q_values=q_values,
                choice=choice,
                r_chosen=r_chosen,
                r_unchosen=r_unchosen,
                params=params,
                counterfactual=counterfactual,
            )
            prev_choice = choice

    return float(-total_ll)


def fit_participant_model(
    df_sub: pd.DataFrame,
    model_name: str = "confirmation",
    counterfactual: bool = True,
    n_restarts: int = 3,
    seed: int = 0,
) -> dict:
    """Fit a specified model to one participant's data and return fitted values + BIC."""
    bounds, x0 = get_bounds_and_init(model_name=model_name, counterfactual=counterfactual)
    rng = np.random.default_rng(seed)
    n_trials = len(df_sub)

    best_res = None
    starts = [x0]
    for _ in range(max(0, n_restarts - 1)):
        jitter = np.array([
            rng.uniform(low + 0.1 * (high - low), high - 0.1 * (high - low))
            for low, high in bounds
        ])
        starts.append(jitter)

    for start_x in starts:
        res = minimize(
            fun=lambda th: compute_nll(df_sub, vector_to_params(th, model_name, counterfactual), counterfactual),
            x0=start_x,
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    theta_hat = best_res.x
    nll = float(best_res.fun)
    k = len(theta_hat)
    bic = float(k * np.log(max(1, n_trials)) + 2.0 * nll)

    out = {"model": model_name, "nll": nll, "bic": bic, "k": k, "theta_hat": theta_hat}
    if model_name == "one":
        out.update({"beta": theta_hat[0], "alpha": theta_hat[1]})
    elif model_name == "valence":
        out.update({"beta": theta_hat[0], "alpha_pos": theta_hat[1], "alpha_neg": theta_hat[2]})
    elif model_name == "information":
        out.update({
            "beta": theta_hat[0],
            "alpha_c": theta_hat[1],
            "alpha_u": theta_hat[2] if counterfactual and len(theta_hat) > 2 else theta_hat[1],
        })
    elif model_name == "confirmation":
        out.update({"beta": theta_hat[0], "alpha_con": theta_hat[1], "alpha_dis": theta_hat[2]})
    elif model_name == "full":
        out.update({
            "beta": theta_hat[0],
            "alpha_c_pos": theta_hat[1],
            "alpha_c_neg": theta_hat[2],
            "alpha_u_pos": theta_hat[3] if counterfactual and len(theta_hat) > 3 else theta_hat[1],
            "alpha_u_neg": theta_hat[4] if counterfactual and len(theta_hat) > 4 else theta_hat[2],
        })
    elif model_name == "perseveration":
        out.update({
            "beta": theta_hat[0],
            "alpha_con": theta_hat[1],
            "alpha_dis": theta_hat[2],
            "pi": theta_hat[3],
        })
    return out


def run_parameter_recovery(
    df_sim: pd.DataFrame,
    true_params_df: pd.DataFrame,
    model_name: str = "confirmation",
    counterfactual: bool = True,
    n_restarts: int = 2,
    base_seed: int = 100,
) -> pd.DataFrame:
    """Fit model_name back to simulated cohort df_sim and merge with true_params_df."""
    records = []
    for i, (pid, df_sub) in enumerate(df_sim.groupby("participant_id", sort=True)):
        fit_res = fit_participant_model(
            df_sub=df_sub,
            model_name=model_name,
            counterfactual=counterfactual,
            n_restarts=n_restarts,
            seed=base_seed + i,
        )
        fit_res["participant_id"] = pid
        records.append(fit_res)

    df_rec = pd.DataFrame(records)
    return true_params_df.merge(df_rec, on="participant_id", suffixes=("_true", "_rec"))


def run_model_recovery(
    sim_datasets_by_true_model: dict[str, pd.DataFrame],
    candidate_models: list[str],
    counterfactual: bool = True,
    n_restarts: int = 2,
    base_seed: int = 500,
) -> pd.DataFrame:
    """Given {true_model_name: simulated_df}, fit every candidate model to each subject
    and return a proportion matrix (rows = true model, cols = winning fitted model by min BIC).
    """
    matrix = pd.DataFrame(0.0, index=candidate_models, columns=candidate_models)
    seed_counter = base_seed

    for true_model, df_cohort in sim_datasets_by_true_model.items():
        pids = sorted(df_cohort["participant_id"].unique())
        win_counts = {m: 0 for m in candidate_models}

        for pid in pids:
            df_sub = df_cohort[df_cohort["participant_id"] == pid]
            bics = {}
            for cand in candidate_models:
                seed_counter += 1
                res = fit_participant_model(
                    df_sub=df_sub,
                    model_name=cand,
                    counterfactual=counterfactual,
                    n_restarts=n_restarts,
                    seed=seed_counter,
                )
                bics[cand] = res["bic"]
            winner = min(bics, key=bics.get)
            win_counts[winner] += 1

        n_sub = max(1, len(pids))
        for cand in candidate_models:
            matrix.loc[true_model, cand] = win_counts[cand] / n_sub

    return matrix
