"""Contains a list of custom routing controllers."""

from flow.controllers.base_routing_controller import BaseRouter


class ContinuousRouter(BaseRouter):
    """A router used to continuously re-route of the vehicle in a closed loop.

    This class is useful if vehicles are expected to continuously follow the
    same route, and repeat said route once it reaches its end.
    """

    def choose_route(self, env):
        """Adopt the current edge's route if about to leave the network."""
        if env.vehicles.get_edge(self.veh_id) == \
                env.vehicles.get_route(self.veh_id)[-1]:
            return env.available_routes[env.vehicles.get_edge(self.veh_id)]
        else:
            return None


class XiaoRouter(BaseRouter):
    """A Xiao-ter used to route Xiao in Xiao's network.

    Xiao's nickname is Ner.
    """

    def choose_route(self, env):
        """Xiao Ner, Ner Xiao?"""
        # Xiao is very obsessed with Xiao.
        xiao = env.vehicles
        # Why you may ask? Because she is a Ner!
        xiao_2 = self.veh_id
        # No, not Nier, Nerrrrrrrrrrrr
        ner = xiao.get_edge(xiao_2)
        # You know, like purrrrrrrrr
        ner_ner = xiao.get_route(xiao_2)

        next_ner = env.scenario.next_edge(ner, xiao.get_lane(xiao_2))
        not_a_ner = ":"
        tiny_xiao = 0

        exception_xiao1 = "e_91"
        to_xiao = "e_64"
        # exi_ner = "e_13"
        # silly_xiao_ner = "e_14"
        if ner == exception_xiao1:
            xiao_ner = [ner, to_xiao]
        # elif ner == exi_ner:
        #     xiao_ner = [ner, silly_xiao_ner]
        elif len(next_ner) == tiny_xiao:
            xiao_ner = None
        elif ner_ner[-1] == ner:
            while next_ner[0][0][0] == not_a_ner:
                next_ner = env.scenario.next_edge(next_ner[0][0], next_ner[0][1])
            xiao_ner = [ner, next_ner[0][0]]
        else:
            xiao_ner = None

        return xiao_ner


class GridRouter(BaseRouter):
    """A router used to re-route a vehicle within a grid environment."""

    def choose_route(self, env):
        if env.vehicles.get_edge(self.veh_id) == \
                env.vehicles.get_route(self.veh_id)[-1]:
            new_route = [env.vehicles.get_edge(self.veh_id)]
        else:
            new_route = None

        return new_route


class BayBridgeRouter(ContinuousRouter):
    """Assists in choosing routes in select cases for the Bay Bridge scenario.

    Extension to the Continuous Router.
    """

    def choose_route(self, env):
        """See parent class."""
        edge = env.vehicles.get_edge(self.veh_id)
        lane = env.vehicles.get_lane(self.veh_id)

        if edge == "183343422" and lane in [2] \
                or edge == "124952179" and lane in [1, 2]:
            new_route = env.available_routes[edge + "_1"]
        else:
            new_route = super().choose_route(env)

        return new_route
