import json
import argparse

import ray
from ray.rllib.agents.agent import get_agent_class
from ray.tune import run_experiments
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env
from flow.utils.rllib import FlowParamsEncoder
from ray.tune import grid_search

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows
from flow.core.traffic_lights import TrafficLights
from flow.core.vehicles import Vehicles
from flow.controllers import RLController, ContinuousRouter

# time horizon of a single rollout
HORIZON = 1500

SCALING = 1
NUM_LANES = 4 * SCALING  # number of lanes in the widest highway
DISABLE_TB = True
DISABLE_RAMP_METER = True
AV_FRAC = 0.10

vehicles = Vehicles()
vehicles.add(
    veh_id="human",
    speed_mode=9,
    routing_controller=(ContinuousRouter, {}),
    lane_change_mode=0,
    num_vehicles=1 * SCALING)
vehicles.add(
    veh_id="rl",
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    speed_mode=9,
    lane_change_mode=0,
    num_vehicles=1)

additional_env_params = {
    "target_velocity": 40,
    "disable_tb": True,
    "disable_ramp_metering": True,
    "controlled_segments": controlled_segments,
    "symmetric": False,
    "observed_segments": num_observed_segments,
    "lane_change_duration": 5,
    "max_accel": 3,
    "max_decel": 3,
}

traffic_lights = TrafficLights()
if not DISABLE_TB:
    traffic_lights.add(node_id="2")
if not DISABLE_RAMP_METER:
    traffic_lights.add(node_id="3")

additional_net_params = {"scaling": SCALING}
net_params = NetParams(
    no_internal_links=False,
    additional_params=additional_net_params)

flow_params = dict(
    # name of the experiment
    exp_tag="augment_bottleneck",

    # name of the flow environment the experiment is running on
    env_name="DesiredVelocityEnv",

    # name of the scenario class the experiment is running on
    scenario="BottleneckScenario",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.5,
        render=False,
        print_warnings=False,
        restart_instance=True,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        warmup_steps=40,
        sims_per_step=1,
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing="uniform",
        min_gap=5,
        lanes_distribution=float("inf"),
        edges_distribution=["2", "3", "4", "5"],
    ),

    # traffic lights to be introduced to specific nodes (see
    # flow.core.traffic_lights.TrafficLights)
    tls=traffic_lights,
)

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="[Flow] Evaluates a Flow Garden solution on a benchmark.",
    epilog=EXAMPLE_USAGE)

# required input parameters
parser.add_argument(
    "--benchmark_name", type=str, help="File path to solution environment.")

# required input parameters
parser.add_argument(
    "--upload_dir", type=str, help="S3 Bucket to upload to.")

# optional input parameters
parser.add_argument(
    '--num_rollouts',
    type=int,
    default=50,
    help="The number of rollouts to train over.")

# optional input parameters
parser.add_argument(
    '--num_cpus',
    type=int,
    default=6,
    help="The number of cpus to use.")

if __name__ == "__main__":
    benchmark_name = 'augment_bottleneck'
    args = parser.parse_args()
    # benchmark name
    benchmark_name = args.benchmark_name
    # number of rollouts per training iteration
    num_rollouts = args.num_rollouts
    # number of parallel workers
    num_cpus = args.num_cpus

    upload_dir = args.upload_dir

    # Import the benchmark and fetch its flow_params
    benchmark = __import__(
        "flow.benchmarks.%s" % benchmark_name, fromlist=["flow_params"])
    flow_params = benchmark.flow_params

    # get the env name and a creator for the environment
    create_env, env_name = make_create_env(params=flow_params, version=0)

    # initialize a ray instance
    ray.init(redirect_output=True)

    alg_run = "ES"

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config["num_workers"] = min(num_cpus, num_rollouts)
    config["episodes_per_batch"] = num_rollouts
    config["eval_prob"] = 0.05
    config["noise_stdev"] = grid_search([0.01, 0.02])
    config["stepsize"] = grid_search([0.01, 0.02])

    config["noise_stdev"] = 0.02
    config["stepsize"] = 0.02

    config["model"]["fcnet_hiddens"] = [100, 50, 25]
    config["observation_filter"] = "NoFilter"
    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                           indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": alg_run,
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 25,
            "max_failures": 999,
            "stop": {"training_iteration": 500},
            "num_samples": 1,
            "upload_dir": "s3://" + upload_dir
        },
    })
