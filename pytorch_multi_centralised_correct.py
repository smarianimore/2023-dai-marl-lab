# Torch
import torch

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.envs.utils import check_env_specs

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm

# Devices
device = "cpu" if not torch.backends.cuda.is_built() else "cuda:0"  # The device where learning is run
vmas_device = device  # The device where the simulator is run (VMAS can run on GPU)

##################################################
##### HYPER-PARAMETERS FOR LEARNING PROCESS ######
##################################################

# Sampling
frames_per_batch = 6_000  # Number of frames collected per training iteration
n_iters = 10  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 30  # Number of optimization steps per training iteration
minibatch_size = 400  # Size of the mini-batches in each optimization step
lr = 3e-4  # Learning rate
max_grad_norm = 1.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.9  # discount factor
lmbda = 0.9  # lambda for generalised advantage estimation
entropy_eps = 1e-4  # coefficient of the entropy term in the PPO loss

########################################
##### CONFIGURATION OF ENVIRONMENT #####
########################################
# In Navigation, randomly spawned agents (circles with surrounding dots) need to navigate to randomly spawned goals
# (smaller circles). Agents need to use LIDARs (dots around them) to avoid colliding into each other. Agents act in a 2D
# continuous world with drag and elastic collisions. Their actions are 2D continuous forces which determine their
# acceleration. The reward is composed of three terms: a collision penalisation, a reward based on the distance to the
# goal, and a final shared reward given when all agents reach their goal. The distance-based term is computed as the
# difference in the relative distance between an agent and its goal over two consecutive timesteps. Each agent observes
# its position, velocity, lidar readings, and relative position to its goal.

max_steps = 100  # Episode steps before done, to avoid infinite episodes
num_vmas_envs = (
    frames_per_batch // max_steps
)  # Number of vectorized envs. frames_per_batch should be divisible by this number
scenario_name = "navigation"
n_agents = 3

env = VmasEnv(
    scenario=scenario_name,
    num_envs=num_vmas_envs,
    continuous_actions=True,  # VMAS supports both continuous and discrete actions
    max_steps=max_steps,
    device=vmas_device,
    # Scenario kwargs
    n_agents=n_agents,  # These are custom kwargs that change for each VMAS scenario, see the VMAS repo to know more.
)
env = TransformedEnv(  # RewardSum transform which will sum rewards over the episode.
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
)
check_env_specs(env)

# SEE RANDOM POLICY IN ACTION
with torch.no_grad():
   env.rollout(
       max_steps=max_steps,
       callback=lambda env, _: env.render(),
       auto_cast_to_device=True,
       break_when_any_done=False,
   )

###################################
##### POLICY NETWORK BUILDING #####
###################################

#  In the navigation environment each agent’s action is represented by a 2-dimensional independent normal distribution.
#  For this, our neural network will have to output a mean and a standard deviation for each action. Each agent will
#  thus have 2 * n_actions_per_agents outputs.

# We need to decide whether we want our agents to share the policy parameters. On the one hand, sharing parameters means
# that they will all share the same policy, which will allow them to benefit from each other’s experiences. This will
# also result in faster training. On the other hand, it will make them behaviorally homogenous, as they will in fact
# share the same model. For this example, we will enable sharing.

share_parameters_policy = True

policy_net = torch.nn.Sequential(
    MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[
            -1
        ],  # n_obs_per_agent
        n_agent_outputs=2 * env.action_spec.shape[-1],  # 2 * n_actions_per_agents
        n_agents=env.n_agents,
        centralised=False,  # each agent will act from its own observation, BUT REMEMBER THAT WE ENABLED PARAMS SHARING!
        share_params=share_parameters_policy,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    ),
    NormalParamExtractor(),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
)

# TorchRL requires wrapping network in a TensorDictModule, a module that will read the in_keys from a tensordict, feed
# them to the neural networks, and write the outputs in-place at the out_keys
policy_module = TensorDictModule(
    policy_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "loc"), ("agents", "scale")],
)

# We now need to build a distribution out of the location and scale of our normal distribution (remember environment
# description). To do so, we instruct the ProbabilisticActor class to build a TanhNormal out of the location and scale
# parameters. We also provide the minimum and maximum values of this distribution, which we gather from the environment
# specs.
policy = ProbabilisticActor(
    module=policy_module,
    spec=env.unbatched_action_spec,
    in_keys=[("agents", "loc"), ("agents", "scale")],
    out_keys=[env.action_key],
    distribution_class=TanhNormal,
    distribution_kwargs={
        "min": env.unbatched_action_spec[env.action_key].space.low,
        "max": env.unbatched_action_spec[env.action_key].space.high,
    },
    return_log_prob=True,  # we'll need the log-prob for the PPO loss
    log_prob_key=("agents", "sample_log_prob"),
)

###################################
##### CRITIC NETWORK BUILDING #####
###################################
# The critic network is a crucial component of the PPO algorithm. This module will read the observations and return the
# corresponding value estimates. As before, one should think carefully about the decision of sharing the critic
# parameters. Sharing is not recommended when agents have different reward functions, as the critics will need to learn
# to assign different values to the same state. In decentralised training settings, sharing cannot be performed without
# additional infrastructure to synchronise parameters.
share_parameters_critic = True
mappo = True  # -> IPPO if False (MAPPO = centralised critic with full obs, IPPO = decentralised critic)

critic_net = MultiAgentMLP(
    n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
    n_agent_outputs=1,  # 1 value per agent (the value estimate for a given state)
    n_agents=env.n_agents,
    centralised=mappo,
    share_params=share_parameters_critic,
    device=device,
    depth=2,
    num_cells=256,
    activation_class=torch.nn.Tanh,
)

critic = TensorDictModule(
    module=critic_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "state_value")],
)

# From this point on, the multi-agent-specific components have been instantiated, and we will simply use the same
# components as in single-agent learning (NICE BENEFIT OF USING WELL-DESIGNED LIBRARIES)

############################
##### LEARNING PROCESS #####
############################

##### CONFIGURATION #####
# Collector classes execute three operations: reset an environment, compute an action using the policy and the latest
# observation, execute a step in the environment, and repeat the last two steps until the environment signals a stop
# (or reaches a done state).
collector = SyncDataCollector(
    env,
    policy,
    device=vmas_device,
    storing_device=device,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
)

# A replay buffer is refilled every time a batch of data is collected, and its data is repeatedly consumed for a certain
# number of epochs. Using a replay buffer for PPO is not mandatory and we could simply use the collected data online,
# but is useful for better reproducibility and generalisation (can switch to off-policy and keep learning pipeline as is).
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(
        frames_per_batch, device=device
    ),  # We store the frames_per_batch collected at each iteration
    sampler=SamplerWithoutReplacement(),  # just use the data as it comes from the environment (state, reward)
    batch_size=minibatch_size,  # We will sample minibatches of this size
)

# PPO requires some “advantage estimation” to be computed. In short, an advantage is a value that reflects an expectancy
# over the return value while dealing with the bias / variance tradeoff (similar to exploration/exploitation).
# To compute the advantage, one just needs to (1) build the advantage module, which utilises our value operator,
# and (2) pass each batch of data through it before each epoch. The GAE module will update the input TensorDict with
# new "advantage" and "value_target" entries. The "value_target" is a gradient-free tensor that represents the
# empirical value that the value network should represent with the input observation.
loss_module = ClipPPOLoss(
    actor=policy,
    critic=critic,
    clip_epsilon=clip_epsilon,
    entropy_coef=entropy_eps,
    normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
)
loss_module.set_keys(  # We have to tell the loss where to find the keys
    reward=env.reward_key,
    action=env.action_key,
    sample_log_prob=("agents", "sample_log_prob"),
    value=("agents", "state_value"),
    # These last 2 keys will be expanded to match the reward shape
    done=("agents", "done"),
    terminated=("agents", "terminated"),
)

loss_module.make_value_estimator(
    ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
)
GAE = loss_module.value_estimator

optim = torch.optim.Adam(loss_module.parameters(), lr)

##### ACTUAL TRAINING STARTS HERE #####
# Core loop is:
#   Collect data
#     Compute advantage
#       Loop over epochs
#         Loop over minibatches to compute loss values
#           Back propagate
#           Optimise
pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")
episode_reward_mean_list = []
for tensordict_data in collector:
    tensordict_data.set(
        ("next", "agents", "done"),
        tensordict_data.get(("next", "done"))
        .unsqueeze(-1)  # https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),  # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
    )
    tensordict_data.set(
        ("next", "agents", "terminated"),
        tensordict_data.get(("next", "terminated"))
        .unsqueeze(-1)
        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
    )

    with torch.no_grad():  # Compute GAE and add it to the data
        GAE(
            tensordict_data,
            params=loss_module.critic_params,
            target_params=loss_module.target_critic_params,
        )

    data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
    replay_buffer.extend(data_view)

    for _ in range(num_epochs):
        for _ in range(frames_per_batch // minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                loss_module.parameters(), max_grad_norm
            )  # Optional

            optim.step()
            optim.zero_grad()

    collector.update_policy_weights_()

    # Logging
    done = tensordict_data.get(("next", "agents", "done"))
    episode_reward_mean = (
        tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
    )
    episode_reward_mean_list.append(episode_reward_mean)
    pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
    pbar.update()

###########################
##### LEARNING CURVES #####
###########################
plt.plot(episode_reward_mean_list)
plt.xlabel("Training iterations")
plt.ylabel("Reward")
plt.title("Episode reward mean")
plt.show()

###############################
##### RENDERLEARNT POLICY #####
###############################
with torch.no_grad():
   env.rollout(
       max_steps=max_steps,
       policy=policy,
       callback=lambda env, _: env.render(),
       auto_cast_to_device=True,
       break_when_any_done=False,
   )
