"""Contains the figure eight scenario class."""

import numpy as np
from numpy import pi, sin, cos, linspace, ceil, sqrt

from flow.core.params import InitialConfig
from flow.core.traffic_lights import TrafficLights
from flow.scenarios.base_scenario import Scenario

ADDITIONAL_NET_PARAMS = {
    # radius of the circular components
    "radius_ring": 30,
    # number of lanes
    "lanes": 1,
    # speed limit for all edges
    "speed_limit": 30,
    # resolution of the curved portions
    "resolution": 40,
    # number of figure 8s
    "num_rings": 7
}

VEHICLE_LENGTH = 5  # length of vehicles in the network, in meters


class MultiFigure8Scenario(Scenario):
    """Figure eight scenario class."""

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLights()):
        """Initialize a figure 8 scenario.

        Requires from net_params:
        - ring_radius: radius of the circular portions of the network. Also
          corresponds to half the length of the perpendicular straight lanes.
        - resolution: number of nodes resolution in the circular portions
        - lanes: number of lanes in the network
        - speed: max speed of vehicles in the network

        In order for right-of-way dynamics to take place at the intersection,
        set "no_internal_links" in net_params to False.

        See flow/scenarios/base_scenario.py for description of params.
        """
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        ring_radius = net_params.additional_params["radius_ring"]
        self.ring_edgelen = ring_radius * np.pi / 2.
        self.intersection_len = 2 * ring_radius
        self.junction_len = 2.9 + 3.3 * net_params.additional_params["lanes"]
        self.inner_space_len = 0.28

        # instantiate "length" in net params
        net_params.additional_params["length"] = \
            6 * self.ring_edgelen + 2 * self.intersection_len + \
            2 * self.junction_len + 10 * self.inner_space_len

        self.radius_ring = net_params.additional_params["radius_ring"]
        self.length = net_params.additional_params["length"]
        self.lanes = net_params.additional_params["lanes"]
        self.resolution = net_params.additional_params["resolution"]
        self.num_rings = net_params.additional_params["num_rings"]

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]
        ring_num = net_params.additional_params["num_rings"]
        ring_spacing = 4 * r
        num_rows = num_cols = int(ceil(sqrt(ring_num)))

        nodes = []
        i = 0
        for j in range(num_rows):
            for k in range(num_cols):
                nodes += [{
                    "id": "center_intersection_{}".format(i),
                    "x": repr(0 + j * ring_spacing),
                    "y": repr(0 + k * ring_spacing),
                    "type": "priority"
                }, {
                    "id": "top_upper_ring_{}".format(i),
                    "x": repr(r + j * ring_spacing),
                    "y": repr(2 * r + k * ring_spacing),
                    "type": "priority"
                }, {
                    "id": "bottom_upper_ring_in_{}".format(i),
                    "x": repr(r + j * ring_spacing),
                    "y": repr(0 + k * ring_spacing),
                    "type": "priority"
                }, {
                    "id": "left_upper_ring_{}".format(i),
                    "x": repr(0 + j * ring_spacing),
                    "y": repr(r + k * ring_spacing),
                    "type": "priority"
                }, {
                    "id": "right_upper_ring_{}".format(i),
                    "x": repr(2 * r + j * ring_spacing),
                    "y": repr(r + k * ring_spacing),
                    "type": "priority"
                }, {
                    "id": "top_lower_ring_{}".format(i),
                    "x": repr(-r + j * ring_spacing),
                    "y": repr(0 + k * ring_spacing),
                    "type": "priority"
                }, {
                    "id": "bottom_lower_ring_{}".format(i),
                    "x": repr(-r + j * ring_spacing),
                    "y": repr(-2 * r + k * ring_spacing),
                    "type": "priority"
                }, {
                    "id": "left_lower_ring_{}".format(i),
                    "x": repr(-2 * r + j * ring_spacing),
                    "y": repr(-r + k * ring_spacing),
                    "type": "priority"
                }, {
                    "id": "right_lower_ring_in_{}".format(i),
                    "x": repr(0 + j * ring_spacing),
                    "y": repr(-r + k * ring_spacing),
                    "type": "priority"
                }]
                i += 1
                if i >= ring_num:
                    break
            if i >= ring_num:
                break
        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        r = net_params.additional_params["radius_ring"]
        resolution = net_params.additional_params["resolution"]
        ring_num = net_params.additional_params["num_rings"]
        num_rows = num_cols = int(ceil(sqrt(ring_num)))
        ring_edgelen = r * pi / 2.
        intersection_edgelen = 2 * r
        ring_spacing = 4 * r
        i = 0
        edges = []

        # intersection edges
        for j in range(num_rows):
            for k in range(num_cols):
                edges += [{
                    "id": "right_lower_ring_in_{}".format(i),
                    "type": "edgeType",
                    "priority": "78",
                    "from": "right_lower_ring_in_{}".format(i),
                    "to": "center_intersection_{}".format(i),
                    "length": repr(intersection_edgelen / 2)
                }, {
                    "id": "right_lower_ring_out_{}".format(i),
                    "type": "edgeType",
                    "priority": "78",
                    "from": "center_intersection_{}".format(i),
                    "to": "left_upper_ring_{}".format(i),
                    "length": repr(intersection_edgelen / 2)
                }, {
                    "id": "bottom_upper_ring_in_{}".format(i),
                    "type": "edgeType",
                    "priority": "46",
                    "from": "bottom_upper_ring_in_{}".format(i),
                    "to": "center_intersection_{}".format(i),
                    "length": repr(intersection_edgelen / 2)
                }, {
                    "id": "bottom_upper_ring_out_{}".format(i),
                    "type": "edgeType",
                    "priority": "46",
                    "from": "center_intersection_{}".format(i),
                    "to": "top_lower_ring_{}".format(i),
                    "length": repr(intersection_edgelen / 2)
                }]

                # ring edges
                edges += [{
                    "id":
                    "left_upper_ring_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "left_upper_ring_{}".format(i),
                    "to":
                    "top_upper_ring_{}".format(i),
                    "length":
                    repr(ring_edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (r * (1 - cos(t)) + j * ring_spacing,
                                        r * (1 + sin(t)) + k * ring_spacing)
                        for t in linspace(0, pi / 2, resolution)
                    ])
                }, {
                    "id":
                    "top_upper_ring_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "top_upper_ring_{}".format(i),
                    "to":
                    "right_upper_ring_{}".format(i),
                    "length":
                    repr(ring_edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (r * (1 + sin(t)) + j * ring_spacing,
                                        r * (1 + cos(t)) + k * ring_spacing)
                        for t in linspace(0, pi / 2, resolution)
                    ])
                }, {
                    "id":
                    "right_upper_ring_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "right_upper_ring_{}".format(i),
                    "to":
                    "bottom_upper_ring_in_{}".format(i),
                    "length":
                    repr(ring_edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (r * (1 + cos(t)) + j * ring_spacing,
                                        r * (1 - sin(t)) + k * ring_spacing)
                        for t in linspace(0, pi / 2, resolution)
                    ])
                }, {
                    "id":
                    "top_lower_ring_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "top_lower_ring_{}".format(i),
                    "to":
                    "left_lower_ring_{}".format(i),
                    "length":
                    repr(ring_edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (-r + r * cos(t) + j * ring_spacing,
                                       -r + r * sin(t) + k * ring_spacing)
                        for t in linspace(pi / 2, pi, resolution)
                    ])
                }, {
                    "id":
                    "left_lower_ring_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "left_lower_ring_{}".format(i),
                    "to":
                    "bottom_lower_ring_{}".format(i),
                    "length":
                    repr(ring_edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (-r + r * cos(t) + j * ring_spacing,
                                       -r + r * sin(t) + k * ring_spacing)
                        for t in linspace(pi, 3 * pi / 2, resolution)
                    ])
                }, {
                    "id":
                    "bottom_lower_ring_{}".format(i),
                    "type":
                    "edgeType",
                    "from":
                    "bottom_lower_ring_{}".format(i),
                    "to":
                    "right_lower_ring_in_{}".format(i),
                    "length":
                    repr(ring_edgelen),
                    "shape":
                    " ".join([
                        "%.2f,%.2f" % (-r + r * cos(t) + j * ring_spacing,
                                        -r + r * sin(t) + k * ring_spacing)
                        for t in linspace(-pi / 2, 0, resolution)
                    ])
                }]
                i += 1
                if i >= ring_num:
                    break
            if i >= ring_num:
                break

        return edges

    def specify_types(self, net_params):
        """See parent class."""
        lanes = net_params.additional_params["lanes"]
        speed_limit = net_params.additional_params["speed_limit"]
        types = [{
            "id": "edgeType",
            "numLanes": repr(lanes),
            "speed": repr(speed_limit)
        }]

        return types

    def specify_routes(self, net_params):
        """See parent class."""
        base_rts = {
            "bottom_lower_ring": [
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring"
            ],
            "right_lower_ring_in": [
                "right_lower_ring_in", "right_lower_ring_out",
                "left_upper_ring", "top_upper_ring", "right_upper_ring",
                "bottom_upper_ring_in", "bottom_upper_ring_out",
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring"
            ],
            "right_lower_ring_out": [
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in"
            ],
            "left_upper_ring": [
                "left_upper_ring", "top_upper_ring", "right_upper_ring",
                "bottom_upper_ring_in", "bottom_upper_ring_out",
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring",
                "right_lower_ring_in", "right_lower_ring_out"
            ],
            "top_upper_ring": [
                "top_upper_ring", "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring"
            ],
            "right_upper_ring": [
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring"
            ],
            "bottom_upper_ring_in": [
                "bottom_upper_ring_in", "bottom_upper_ring_out",
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring",
                "right_lower_ring_in", "right_lower_ring_out",
                "left_upper_ring", "top_upper_ring", "right_upper_ring"
            ],
            "bottom_upper_ring_out": [
                "bottom_upper_ring_out", "top_lower_ring", "left_lower_ring",
                "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in"
            ],
            "top_lower_ring": [
                "top_lower_ring", "left_lower_ring", "bottom_lower_ring",
                "right_lower_ring_in", "right_lower_ring_out",
                "left_upper_ring", "top_upper_ring", "right_upper_ring",
                "bottom_upper_ring_in", "bottom_upper_ring_out"
            ],
            "left_lower_ring": [
                "left_lower_ring", "bottom_lower_ring", "right_lower_ring_in",
                "right_lower_ring_out", "left_upper_ring", "top_upper_ring",
                "right_upper_ring", "bottom_upper_ring_in",
                "bottom_upper_ring_out", "top_lower_ring"
            ]
        }
        rts = {}

        for i in range(self.num_rings):
            for base_edge in base_rts:
                base_rt = base_rts[base_edge]
                edge = base_edge + "_{}".format(i)
                edge_rts = [rt_edge + "_{}".format(i) for rt_edge in base_rt]
                rts[edge] = edge_rts

        return rts

    def specify_edge_starts(self):
        """See base class."""
        edgestarts = []

        for i in range(self.num_rings):
            edgestarts += \
                [("bottom_lower_ring_{}".format(i),
                  0 + self.inner_space_len + i * self.length),
                 ("right_lower_ring_in_{}".format(i),
                  self.ring_edgelen + 2 * self.inner_space_len + i * self.length),
                 ("right_lower_ring_out_{}".format(i),
                  self.ring_edgelen + self.intersection_len / 2 +
                  self.junction_len + 3 * self.inner_space_len + i * self.length),
                 ("left_upper_ring_{}".format(i),
                  self.ring_edgelen + self.intersection_len +
                  self.junction_len + 4 * self.inner_space_len + i * self.length),
                 ("top_upper_ring_{}".format(i),
                  2 * self.ring_edgelen + self.intersection_len +
                  self.junction_len + 5 * self.inner_space_len + i * self.length),
                 ("right_upper_ring_{}".format(i),
                  3 * self.ring_edgelen + self.intersection_len +
                  self.junction_len + 6 * self.inner_space_len + i * self.length),
                 ("bottom_upper_ring_in_{}".format(i),
                  4 * self.ring_edgelen + self.intersection_len +
                  self.junction_len + 7 * self.inner_space_len + i * self.length),
                 ("bottom_upper_ring_out_{}".format(i),
                  4 * self.ring_edgelen + 3 / 2 * self.intersection_len +
                  2 * self.junction_len + 8 * self.inner_space_len + i * self.length),
                 ("top_lower_ring_{}".format(i),
                  4 * self.ring_edgelen + 2 * self.intersection_len +
                  2 * self.junction_len + 9 * self.inner_space_len + i * self.length),
                 ("left_lower_ring_{}".format(i),
                  5 * self.ring_edgelen + 2 * self.intersection_len +
                  2 * self.junction_len + 10 * self.inner_space_len + i * self.length)]

        return edgestarts

    def specify_intersection_edge_starts(self):
        """See base class."""
        intersection_edgestarts = []
        for i in range(self.num_rings):
            intersection_edgestarts += \
                [(":center_intersection_"+str(i)+"_%s" % (1 + self.lanes),
                  self.ring_edgelen + self.intersection_len / 2 +
                  3 * self.inner_space_len + i * self.length),
                 (":center_intersection_"+str(i)+"_1",
                  4 * self.ring_edgelen + 3 / 2 * self.intersection_len +
                  self.junction_len + 8 * self.inner_space_len)]
        return intersection_edgestarts

    def specify_internal_edge_starts(self):
        """See base class."""
        internal_edgestarts = []
        for i in range(self.num_rings):
            internal_edgestarts += \
                [(":bottom_lower_ring_{}".format(i),
                  0 + i * self.length),
                 (":right_lower_ring_in_{}".format(i),
                  self.ring_edgelen +
                  self.inner_space_len + i * self.length),
                 (":right_lower_ring_out_{}".format(i),
                  self.ring_edgelen + self.intersection_len / 2 +
                  self.junction_len +
                  2 * self.inner_space_len + i * self.length),
                 (":left_upper_ring_{}".format(i),
                  self.ring_edgelen + self.intersection_len +
                  self.junction_len +
                  3 * self.inner_space_len + i * self.length),
                 (":top_upper_ring_{}".format(i),
                  2 * self.ring_edgelen + self.intersection_len +
                  self.junction_len +
                  4 * self.inner_space_len + i * self.length),
                 (":right_upper_ring_{}".format(i),
                  3 * self.ring_edgelen + self.intersection_len +
                  self.junction_len +
                  5 * self.inner_space_len + i * self.length),
                 (":bottom_upper_ring_in_{}".format(i),
                  4 * self.ring_edgelen + self.intersection_len +
                  self.junction_len +
                  6 * self.inner_space_len + i * self.length),
                 (":bottom_upper_ring_out_{}".format(i),
                  4 * self.ring_edgelen + 3 / 2 * self.intersection_len +
                  2 * self.junction_len +
                  7 * self.inner_space_len + i * self.length),
                 (":top_lower_ring_{}".format(i),
                  4 * self.ring_edgelen + 2 * self.intersection_len +
                  2 * self.junction_len +
                  8 * self.inner_space_len + i * self.length),
                 (":left_lower_ring_{}".format(i),
                  5 * self.ring_edgelen + 2 * self.intersection_len +
                  2 * self.junction_len +
                  9 * self.inner_space_len + i * self.length)]
        return internal_edgestarts


    def gen_custom_start_pos(self, initial_config, num_vehicles, **kwargs):
        """Generate uniformly spaced starting positions.

        It is assumed that there are an equal number of vehicles per ring.
        If the perturbation term in initial_config is set to some positive
        value, then the start positions are perturbed from a uniformly spaced
        distribution by a gaussian whose std is equal to this perturbation
        term.

        Parameters
        ----------
        initial_config : InitialConfig type
            see flow/core/params.py
        num_vehicles : int
            number of vehicles to be placed on the network
        kwargs : dict
            extra components, usually defined during reset to overwrite initial
            config parameters

        Returns
        -------
        startpositions : list of tuple (float, float)
            list of start positions [(edge0, pos0), (edge1, pos1), ...]
        startlanes : list of int
            list of start lanes
        """
        (x0, min_gap, bunching, lanes_distr, available_length,
         available_edges, initial_config) = \
            self._get_start_pos_util(initial_config, num_vehicles, **kwargs)

        # return an empty list of starting positions and lanes if there are no
        # vehicles to be placed
        if num_vehicles == 0:
            return [], []

        increment = available_length / num_vehicles
        vehs_per_ring = num_vehicles / self.num_rings

        x = x0
        car_count = 0
        startpositions, startlanes = [], []

        # generate uniform starting positions
        while car_count < num_vehicles:
            # collect the position and lane number of each new vehicle
            pos = self.get_edge(x)

            # ensures that vehicles are not placed in an internal junction
            while pos[0] in dict(self.internal_edgestarts).keys():
                # find the location of the internal edge in total_edgestarts,
                # which has the edges ordered by position
                edges = [tup[0] for tup in self.total_edgestarts]
                indx_edge = next(
                    i for i, edge in enumerate(edges) if edge == pos[0])

                # take the next edge in the list, and place the car at the
                # beginning of this edge
                if indx_edge == len(edges) - 1:
                    next_edge_pos = self.total_edgestarts[0]
                else:
                    next_edge_pos = self.total_edgestarts[indx_edge + 1]

                x = next_edge_pos[1]
                pos = (next_edge_pos[0], 0)

            # ensures that you are in an acceptable edge
            while pos[0] not in available_edges:
                x = (x + self.edge_length(pos[0]))
                pos = self.get_edge(x)

            # place vehicles side-by-side in all available lanes on this edge
            for lane in range(min([self.num_lanes(pos[0]), lanes_distr])):
                car_count += 1
                startpositions.append(pos)
                startlanes.append(lane)

                if car_count == num_vehicles:
                    break

            x = (x + increment + VEHICLE_LENGTH + min_gap) + 1e-13

            if (car_count % vehs_per_ring) == 0:
                # if we have put in the right number of cars,
                # move onto the next ring
                ring_num = int(car_count / vehs_per_ring)
                x = self.length * ring_num + 1e-13

        return startpositions, startlanes
