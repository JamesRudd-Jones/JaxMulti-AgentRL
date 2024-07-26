import sys
import os
import importlib
import jax.numpy as jnp
from typing import NamedTuple, Any, Mapping
import chex
import jax


class MemoryState(NamedTuple):
    hstate: jnp.ndarray
    extras: Mapping[str, jnp.ndarray]


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mem_state: MemoryState
    info: jnp.ndarray


class EvalTransition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    distribution: Any
    spec_key: chex.PRNGKey
    env_state: jnp.ndarray


def import_class_from_folder(folder_name):
    """
    Imports a class from a folder with the same name

    Args:
        folder_name (str): The name of the folder and potential class.

    Returns:
        The imported class, or None if import fails.
    """

    if not isinstance(folder_name, str):
        raise TypeError("folder_name must be a string.")

    # Check for multiple potential entries
    potential_path = os.path.join(os.curdir, "project_name", "agents",
                                  folder_name)  # TODO the project_name addition ain't great

    if os.path.isdir(potential_path) and os.path.exists(
            os.path.join(potential_path, f"{folder_name}.py")):
        # Use importlib to dynamically import the module
        module_spec = importlib.util.spec_from_file_location(folder_name,
                                                             os.path.join(potential_path, f"{folder_name}.py"))
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        # Retrieve the class from the imported module
        return getattr(module, f"{folder_name}Agent")

    else:
        print(f"Error: Folder '{folder_name}' not found in any search paths.")
        return None


def batchify(x: dict, agent_list, num_agents, num_envs):
    inter = jnp.stack([x[a] for a in agent_list])
    return inter.reshape((num_agents, num_envs, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_agents, num_devices):
    x = x.reshape((num_agents, num_devices, -1))
    return {i: x[i] for i in agent_list}


def ipd_visitation(observations: jnp.ndarray, actions: jnp.ndarray, final_obs: jnp.ndarray) -> dict:
    observations = observations[:, 0, :][:, jnp.newaxis, :]  # TODO index to agent 0 again
    actions = actions[:, 0, :][:, jnp.newaxis, :]  # TODO index to agent 0 again
    final_obs = final_obs[0]  # TODO index to agent 0

    # obs [num_inner_steps, num_agents, num_envs, ...]
    # final_t [num_opps, num_envs, ...]
    num_timesteps = observations.shape[0]
    # obs = [0, 1, 2, 3, 4], a = [0, 1]
    # combine = [0, .... 9]
    state_actions = 2 * jnp.argmax(observations, axis=-1) + actions
    state_actions = jnp.reshape(state_actions, (num_timesteps,) + state_actions.shape[1:],)
    # assume final step taken is cooperate

    final_obs = jax.lax.expand_dims(2 * jnp.argmax(final_obs, axis=-1), [0])
    state_actions = jnp.append(state_actions, final_obs[jnp.newaxis, :, :], axis=0)
    hist = jnp.bincount(state_actions.flatten(), length=10)
    state_freq = hist.reshape((int(hist.shape[0] / 2), 2)).sum(axis=1)
    state_probs = state_freq / state_freq.sum()
    action_probs = jnp.nan_to_num(hist[::2] / state_freq)
    return {
        "state_visitation/CC": state_freq[0],
        "state_visitation/CD": state_freq[1],
        "state_visitation/DC": state_freq[2],
        "state_visitation/DD": state_freq[3],
        "state_visitation/START": state_freq[4],
        "state_probability/CC": state_probs[0],
        "state_probability/CD": state_probs[1],
        "state_probability/DC": state_probs[2],
        "state_probability/DD": state_probs[3],
        "state_probability/START": state_probs[4],
        "cooperation_probability/CC": action_probs[0],
        "cooperation_probability/CD": action_probs[1],
        "cooperation_probability/DC": action_probs[2],
        "cooperation_probability/DD": action_probs[3],
        "cooperation_probability/START": action_probs[4],
    }
