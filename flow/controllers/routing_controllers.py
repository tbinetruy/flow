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
        if len(env.vehicles.get_route(self.veh_id)) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif env.vehicles.get_edge(self.veh_id) == \
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
        cur_route = env.vehicles.get_route(self.veh_id)
        route_assigned = False

        if len(cur_route) > 1:
            route_assigned = True
        if 'section1' in type_id and not route_assigned:
            route = ['e_2', 'e_1', 'e_7', 'e_8_b', 'e_8_u', 'e_9', 'e_10', 'e_11']
        elif 'section2' in type_id and not route_assigned:
            route = ['e_3', 'e_25', 'e_30', 'e_31', 'e_32', 'e_21', 'e_8_u']
        elif 'section3' in type_id and not route_assigned:
            route = ['e_41', 'e_39', 'e_37', 'e_29_u', 'e_21', 'e_8_u', 'e_9']
        elif 'section4' in type_id and not route_assigned:
            route = ['e_39', 'e_37', 'e_29_u', 'e_21']
        elif 'section6' in type_id and not route_assigned:
            route = ['e_60', 'e_69','e_72','e_68','e_66','e_63','e_94','e_52','e_38','e_50','e_60']
        elif 'section5' in type_id and not route_assigned:
            route = ['e_34', 'e_23', 'e_15', 'e_16','e_20', 'e_47', 'e_34']
        elif 'section7' in type_id and not route_assigned:
            route = ['e_42', 'e_44','e_46','e_48','e_78','e_86', 'e_59']
        # elif 'top_center_down' in type_id and not route_assigned:
        #     route = ['e_79','e_47', 'e_45']
        elif 'section8' in type_id and not route_assigned:
            route = ['e_73','e_75','e_77','e_81','e_84','e_85','e_90']
        elif 'idm' in type_id:
            route = MinicityRouter.choose_route(self, env)
        else:
            route = None

        return route


class GridRouter(BaseRouter):
    """A router used to re-route a vehicle within a grid environment."""

    def choose_route(self, env):
        if len(env.vehicles.get_route(self.veh_id)) == 0:
            # this occurs to inflowing vehicles, whose information is not added
            # to the subscriptions in the first step that they departed
            return None
        elif env.vehicles.get_edge(self.veh_id) == \
                env.vehicles.get_route(self.veh_id)[-1]:
            return [env.vehicles.get_edge(self.veh_id)]
        else:
            return None


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
