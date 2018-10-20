"""Example of modified minicity network with human-driven vehicles.
"""
from flow.controllers import IDMController, StaticLaneChanger, ContinuousRouter, \
    RLController
from flow.core.experiment import SumoExperiment
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig
from flow.core.vehicles import Vehicles
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.scenarios.minicity.scenario import MiniCityScenario, \
    ADDITIONAL_NET_PARAMS
from flow.scenarios.minicity.gen import MiniCityGenerator
from flow.controllers.routing_controllers import XiaoRouter


def minicity_example(render=None):
    """
    Perform a simulation of vehicles on modified minicity of University of Delaware.

    Parameters
    ----------
    render: bool, optional
        specifies whether to use sumo's gui during execution

    Returns
    -------
    exp: flow.core.SumoExperiment type
        A non-rl experiment demonstrating the performance of human-driven
        vehicles on the minicity scenario.
    """
    sumo_params = SumoParams(render=True)

    if render is not None:
        sumo_params.render = render

    vehicles = Vehicles()
    vehicles.add(
        veh_id="idm",
        acceleration_controller=(IDMController, {}),
        lane_change_controller=(StaticLaneChanger, {}),
        routing_controller=(XiaoRouter, {}),
        speed_mode="no_collide",
        initial_speed=0,
        num_vehicles=30)
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        lane_change_controller=(StaticLaneChanger, {}),
        routing_controller=(XiaoRouter, {}),
        speed_mode="no_collide",
        initial_speed=0,
        num_vehicles=10)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    net_params = NetParams(
        no_internal_links=False, additional_params=additional_net_params)

    initial_config = InitialConfig(spacing="random")

    scenario = MiniCityScenario(
        name="minicity",
        generator_class=MiniCityGenerator,
        vehicles=vehicles,
        initial_config=initial_config,
        net_params=net_params)

    env = AccelEnv(env_params, sumo_params, scenario)

    return SumoExperiment(env, scenario)


if __name__ == "__main__":
    # import the experiment variable
    exp = minicity_example(render=False)

    # run for a set number of rollouts / time steps
    exp.run(1, 1500)
