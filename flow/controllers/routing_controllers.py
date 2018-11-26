import random

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


class MinicityRouter(BaseRouter):
    """A router used to continuously re-route vehicles in minicity scenario.

    This class allows the vehicle to pick a random route at junctions.
    """

    def choose_route(self, env):
        """See parent class."""
        vehicles = env.vehicles
        veh_id = self.veh_id
        veh_edge = vehicles.get_edge(veh_id)
        veh_route = vehicles.get_route(veh_id)
        veh_next_edge = env.scenario.next_edge(veh_edge,
                                               vehicles.get_lane(veh_id))
        not_an_edge = ":"
        no_next = 0

        if len(veh_next_edge) == no_next:
            next_route = None
        elif veh_route[-1] == veh_edge:
            random_route = random.randint(0, len(veh_next_edge) - 1)
            while veh_next_edge[0][0][0] == not_an_edge:
                veh_next_edge = env.scenario.next_edge(
                    veh_next_edge[random_route][0],
                    veh_next_edge[random_route][1])
            next_route = [veh_edge, veh_next_edge[0][0]]
        else:
            next_route = None

        if veh_edge in ['e_37', 'e_51']:
            next_route = [veh_edge, 'e_29_u', 'e_21']

        return next_route


class MinicityTrainingRouter(MinicityRouter):

    def choose_route(self, env):
        type_id = env.vehicles.get_state(self.veh_id, 'type')
        edge = env.vehicles.get_edge(self.veh_id)
        cur_route = env.vehicles.get_route(self.veh_id)
        # print(cur_route)
        if 'bottom_left' in type_id and cur_route[0] == edge:
            # print(self.veh_id, edge)
            route = ['e_7', 'e_8_b', 'e_8_u', 'e_9', 'e_10', 'e_11', 'e_12']
        elif 'center_left' in type_id and cur_route[0] == edge:
            # print(self.veh_id, edge)
            route = ['e_25', 'e_30', 'e_31', 'e_32', 'e_21', 'e_8_u', 'e_9']
        elif 'center_center' in type_id and cur_route[0] == edge:
            # print(self.veh_id, edge)
            route = ['e_41', 'e_39', 'e_37', 'e_29_u', 'e_21', 'e_8_u', 'e_9']
        elif 'bottom_center' in type_id and cur_route[0] == edge:
            # print(self.veh_id, edge)
            route = ['e_38', 'e_88', 'e_26', 'e_2', 'e_1', 'e_7']
        elif env.vehicles.get_edge(self.veh_id) == \
                env.vehicles.get_route(self.veh_id)[-1]:
            route = None
        else:
            route = MinicityRouter.choose_route(self, env)

        return route


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
