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
    InFlows, SumoCarFollowingParams
from flow.core.vehicles import Vehicles
from flow.controllers import SumoCarFollowingController, GridRouter

# time horizon of a single rollout
HORIZON = 400
# inflow rate of vehicles at every edge
EDGE_INFLOW = 300
# enter speed for departing vehicles
V_ENTER = 30
# number of row of bidirectional lanes
N_ROWS = 1
# number of columns of bidirectional lanes
N_COLUMNS = 1
# length of inner edges in the grid network
INNER_LENGTH = 90
# length of final edge in route
LONG_LENGTH = 100
# length of edges that vehicles start on
SHORT_LENGTH = 90
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 10, 10, 10, 10

# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = Vehicles()
vehicles.add(
    veh_id="human",
    acceleration_controller=(SumoCarFollowingController, {}),
    sumo_car_following_params=SumoCarFollowingParams(
        max_speed=V_ENTER,
    ),
    routing_controller=(GridRouter, {}),
    num_vehicles=(N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS,
    speed_mode="no_collide")  # Choose from "right_of_way" and "no_collide"

# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

flow_params = dict(
    # name of the experiment
    exp_tag="augment_intersection",

    # name of the flow environment the experiment is running on
    env_name="PO_TrafficLightGridEnv",

    # name of the scenario class the experiment is running on
    scenario="SimpleGridScenario",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        restart_instance=True,
        sim_step=1,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": 50,
            "switch_time": 3,
            "num_observed": 2,
            "discrete": False,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params={
            "speed_limit": V_ENTER + 5,
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "horizontal_lanes": 1,
            "vertical_lanes": 1,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(shuffle=True),
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
    benchmark_name = 'augment_intersection'
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
