"""Contains the bottleneck scenario class."""

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.base_scenario import Scenario

ADDITIONAL_NET_PARAMS = {}


class RoundAboutScenario(Scenario):
    """Scenario class for bottleneck simulations."""

    def __init__(self,
                 name,
                 generator_class,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Instantiate the scenario class.

        Requires from net_params:
        - scaling: the factor multiplying number of lanes

        In order for right-of-way dynamics to take place at the intersection,
        set "no_internal_links" in net_params to False.

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, generator_class, vehicles, net_params,
                         initial_config, traffic_lights)

    def specify_edge_starts(self):
        """See parent class."""
        # the total length of the network is defined within this function
        self.length = 0

        edgestarts = []
        for edge_id in self._edge_list:
            # the current edge starts where the last edge ended
            edgestarts.append((edge_id, self.length))
            # increment the total length of the network with the length of the
            # current edge
            self.length += self._edges[edge_id]["length"]

        return edgestarts
