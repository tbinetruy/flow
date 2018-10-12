"""Contains the bottleneck generator class."""

from flow.core.generator import Generator
from numpy import linspace, pi, sin, cos

SCALING = 50


class RoundAboutGenerator(Generator):
    """Generator class for simulating a bottleneck.

    No parameters are needed from net_params (the network is not parametrized).
    """

    def specify_nodes(self, net_params):
        """See parent class."""
        nodes = [{"id": "n_1", "x": -1.0, "y": 0},
                 {"id": "n_2", "x": -0.5, "y": 0},
                 {"id": "n_3", "x": 0, "y": 0.5},
                 {"id": "n_4", "x": 0, "y": 1.0},
                 {"id": "n_5", "x": 0.5, "y": 0},
                 {"id": "n_6", "x": 1.0, "y": 0}
                ]

        for node in nodes:
            node["x"] = str(node["x"] * SCALING)
            node["y"] = str(node["y"] * SCALING)

        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        resolution = 40

        edges = [{"id": "e_1", "from": "n_1", "to": "n_2", "length": None,
                  "numLanes": 2, "type": "edgeType"},
                 {"id": "e_2", "from": "n_3", "to": "n_2", "length": None,
                  "numLanes": 2, "type": "edgeType",
                  "shape": [(0 + 0.5 * cos(t), 0 + 0.5 * sin(t))
                            for t in linspace(pi / 2, pi, resolution)]
                  },
                 {"id": "e_3", "from": "n_3", "to": "n_4", "length": None,
                  "numLanes": 2, "type": "edgeType"},
                 {"id": "e_4", "from": "n_2", "to": "n_5", "length": None,
                  "numLanes": 2, "type": "edgeType",
                  "shape": [(0 + 0.5 * cos(t), 0 + 0.5 * sin(t))
                            for t in linspace(pi, 2 * pi, resolution)]
                  },
                 {"id": "e_5", "from": "n_5", "to": "n_3", "length": None,
                  "numLanes": 2, "type": "edgeType",
                  "shape": [(0 + 0.5 * cos(t), 0 + 0.5 * sin(t))
                            for t in linspace(0, pi / 2, resolution)]
                  },
                 {"id": "e_6", "from": "n_5", "to": "n_6", "length": None,
                  "numLanes": 2, "type": "edgeType"}
                 ]

        for edge in edges:
            edge["length"] = "1"
            edge["numLanes"] = str(edge["numLanes"])
            if "shape" in edge:
                edge["shape"] = " ".join("%.2f,%.2f" % (blip*SCALING, blop*SCALING) for blip, blop in edge["shape"])

        return edges

    # def specify_connections(self, net_params):
    #     """See parent class."""
    #     scaling = net_params.additional_params.get("scaling", 1)
    #     conn = []
    #     for i in range(4 * scaling):
    #         conn += [{
    #             "from": "3",
    #             "to": "4",
    #             "fromLane": str(i),
    #             "toLane": str(int(np.floor(i / 2)))
    #         }]
    #     for i in range(2 * scaling):
    #         conn += [{
    #             "from": "4",
    #             "to": "5",
    #             "fromLane": str(i),
    #             "toLane": str(int(np.floor(i / 2)))
    #         }]
    #     return conn

    def specify_types(self, net_params):
        """See parent class."""
        types = [{"id": "edgeType", "speed": repr(30)}]
        return types

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {"e_1": ["e_1"],
               "e_2": ["e_2"],
               "e_3": ["e_3"],
               "e_4": ["e_4"],
               "e_5": ["e_5"]}

        return rts
