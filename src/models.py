from dataclasses import dataclass
import numpy as np


@dataclass
class RLParams:
    beta: float = 3.0
    alpha_c_pos: float = 0.5
    alpha_c_neg: float = 0.15
    alpha_u_pos: float = 0.15
    alpha_u_neg: float = 0.5
    perseveration: float = 0.0

    @classmethod
    def from_confirmation(
        cls,
        beta: float,
        alpha_con: float,
        alpha_dis: float,
        perseveration: float = 0.0,
    ) -> "RLParams":
        """Confirmation parameterization:
        - Chosen + (PE_c > 0) and Unchosen - (PE_u < 0) use alpha_con
        - Chosen - (PE_c < 0) and Unchosen + (PE_u > 0) use alpha_dis
        """
        return cls(
            beta=beta,
            alpha_c_pos=alpha_con,
            alpha_c_neg=alpha_dis,
            alpha_u_pos=alpha_dis,
            alpha_u_neg=alpha_con,
            perseveration=perseveration,
        )


def softmax_choice_prob(
    q_values: np.ndarray,
    beta: float,
    prev_choice: int | None = None,
    pi: float = 0.0,
) -> np.ndarray:
    """Compute choice probabilities over actions (shape: [n_actions])."""
    logits = beta * np.asarray(q_values, dtype=float)
    if prev_choice is not None and pi != 0.0:
        logits[prev_choice] += pi
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits)


def update_q_values(
    q_values: np.ndarray,
    choice: int,
    r_chosen: float,
    r_unchosen: float | None,
    params: RLParams,
    counterfactual: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Update Q-values for one trial in a 2-option context."""
    q_next = np.array(q_values, dtype=float, copy=True)
    unchosen = 1 - choice

    pe_c = float(r_chosen) - q_next[choice]
    lr_c = params.alpha_c_pos if pe_c >= 0 else params.alpha_c_neg
    q_next[choice] = q_next[choice] + lr_c * pe_c

    info = {"pe_c": pe_c, "lr_c": lr_c}

    if counterfactual and r_unchosen is not None:
        pe_u = float(r_unchosen) - q_next[unchosen]
        lr_u = params.alpha_u_pos if pe_u >= 0 else params.alpha_u_neg
        q_next[unchosen] = q_next[unchosen] + lr_u * pe_u
        info.update({"pe_u": pe_u, "lr_u": lr_u})

    return q_next, info
