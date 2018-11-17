import pyglet
from pyglet.gl import *
from matplotlib import cm
import numpy as np
import cv2, imutils
import os
from os.path import expanduser
HOME = expanduser("~")
import time

class PygletRenderer():

    def __init__(self, kernel, mode,
                 save_render=False,
                 path=HOME+"/flow_rendering",
                 sight_radius=50,
                 pxpm=2,
                 show_radius=False):
        self.kernel = kernel
        self.mode = mode
        if self.mode not in [True, False, "rgb", "drgb", "gray", "dgray"]:
            raise ValueError("Mode %s is not supported!" % self.mode)
        self.save_render = save_render
        if self.save_render:
            if not os.path.exists(path):
                os.mkdir(path)
            self.path = path + '/' + time.strftime("%Y%m%d-%H%M%S")
            os.mkdir(self.path)
        self.sight_radius = sight_radius
        self.pxpm = pxpm # Pixel per meter
        self.show_radius = show_radius
        self.time = 0

        self.lane_polys = []
        lane_polys_flat = []
        for lane_id in self.kernel.lane.getIDList():
            _lane_poly = self.kernel.lane.getShape(lane_id)
            lane_poly = [i for pt in _lane_poly for i in pt]
            self.lane_polys.append(lane_poly)
            lane_polys_flat += lane_poly

        polys_x = np.asarray(lane_polys_flat[::2])
        width = int(polys_x.max() - polys_x.min())
        shift = polys_x.min() - 2
        scale = (width - 4) / width
        self.width = (width + 2*self.sight_radius) * self.pxpm
        self.x_shift = shift - self.sight_radius
        self.x_scale = scale

        polys_y = np.asarray(lane_polys_flat[1::2])
        height = int(polys_y.max() - polys_y.min())
        shift = polys_y.min() - 2
        scale = (height - 4) / height
        self.height = (height + 2*self.sight_radius) * self.pxpm
        self.y_shift = shift - self.sight_radius
        self.y_scale = scale

        self.lane_colors = []
        for lane_poly in self.lane_polys:
            lane_poly[::2] = [(x-self.x_shift)*self.x_scale*self.pxpm
                              for x in lane_poly[::2]]
            lane_poly[1::2] = [(y-self.y_shift)*self.y_scale*self.pxpm
                               for y in lane_poly[1::2]]
            color = [c for _ in range(int(len(lane_poly)/2))
                     for c in [250, 250, 255]]
            self.lane_colors.append(color)

        self.window = pyglet.window.Window(width=self.width,
                                           height=self.height)
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        frame = frame.reshape(buffer.height, buffer.width, 4)
        self.frame = frame[::-1,:,0:3][...,::-1]

        print("Rendering with Pyglet with frame size", (self.width, self.height))


    def render(self, human_orientations, machine_orientations,
               human_dynamics, machine_dynamics,
               colors=None, sight_radius=None, show_radius=None):
        if sight_radius is not None:
            sight_radius = sight_radius * self.pxpm
        else:
            sight_radius = self.sight_radius
        if show_radius is None:
            show_radius = self.show_radius

        self.time = self.kernel.simulation.getCurrentTime()
        self.time /= self.kernel.simulation.getDeltaT()
        self.time = int(self.time)

        glClearColor(0.125,0.125,0.125,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self.lane_batch = pyglet.graphics.Batch()
        self.add_lane_polys()
        self.lane_batch.draw()
        self.vehicle_batch = pyglet.graphics.Batch()
        if "d" in self.mode:
            human_conditions = [
                (255*np.array(cm.summer_r(d)[:3])).astype(np.uint8).tolist()
                for d in human_dynamics]
        else:
            human_conditions = [[0, 128, 128] for d in human_dynamics]
        self.add_vehicle_polys(human_orientations,
                               human_conditions, 0)
        if "d" in self.mode:
            machine_conditions = [
                (255*np.array(cm.spring_r(d)[:3])).astype(np.uint8).tolist()
                for d in machine_dynamics]
        else:
            machine_conditions = [[255, 255, 255] for d in machine_dynamics]
            self.add_vehicle_polys(machine_orientations,
                                   machine_conditions,
                                   sight_radius)
        if show_radius:
            self.add_vehicle_polys(machine_orientations,
                                   machine_conditions,
                                   sight_radius)
        else:
            self.add_vehicle_polys(machine_orientations,
                                   machine_conditions, 0)
        self.vehicle_batch.draw()

        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        frame = frame.reshape(buffer.height, buffer.width, 4)
        self.frame = frame[::-1,:,0:3][...,::-1]
        self.window.flip()

        if "gray" in self.mode:
            _frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        else:
            _frame = self.frame
        if self.save_render:
            cv2.imwrite("%s/frame_%06d.png" % \
                        (self.path, self.time), _frame)
        return _frame


    def get_sight(self, orientation, id, sight_radius=None):
        if sight_radius is not None:
            sight_radius = sight_radius * self.pxpm
        else:
            sight_radius = self.sight_radius * self.pxpm
        x, y, ang = orientation
        x = (x-self.x_shift)*self.x_scale*self.pxpm
        y = (y-self.y_shift)*self.y_scale*self.pxpm
        x_med = x
        y_med = self.height - y # TODO: Check this!
        x_min = int(x_med - sight_radius)
        y_min = int(y_med - sight_radius)
        x_max = int(x_med + sight_radius)
        y_max = int(y_med + sight_radius)
        fixed_sight = self.frame[y_min:y_max, x_min:x_max]
        height, width = fixed_sight.shape[0:2]
        mask = np.zeros((height,width), np.uint8)
        cv2.circle(mask, (int(sight_radius), int(sight_radius)),
                   int(sight_radius), (255,255,255), thickness=-1)
        rotated_sight = cv2.bitwise_and(fixed_sight, fixed_sight, mask=mask)
        rotated_sight = imutils.rotate(rotated_sight, ang)
        if "gray" in self.mode:
            rotated_sight = cv2.cvtColor(rotated_sight, cv2.COLOR_BGR2GRAY)
        if self.save_render:
            cv2.imwrite("%s/sight_%s_%06d.png" % \
                        (self.path, id, self.time), \
                        rotated_sight)
        return rotated_sight


    def close(self):
        self.window.close()


    def add_lane_polys(self):
        for lane_poly, lane_color in zip(self.lane_polys, self.lane_colors):
            self._add_lane_poly(lane_poly, lane_color)


    def _add_lane_poly(self, lane_poly, lane_color):
        num = int(len(lane_poly)/2)
        index = [x for x in range(num)]
        group = pyglet.graphics.Group()
        self.lane_batch.add_indexed(num, GL_LINE_STRIP, group, index,
                               ("v2f", lane_poly), ("c3B", lane_color))


    def add_vehicle_polys(self, orientations, colors, sight_radius):
        for orientation, color in zip(orientations, colors):
            x, y, ang = orientation
            x = (x-self.x_shift)*self.x_scale*self.pxpm
            y = (y-self.y_shift)*self.y_scale*self.pxpm
            self._add_vehicle_poly_triangle((x, y), ang, 5, color)
            self._add_vehicle_poly_circle((x, y), sight_radius, color)


    def _add_vehicle_poly_triangle(self, center, angle, size, color):
        cx, cy = center
        ang = np.radians(angle)
        s = size*self.pxpm
        pt1 = [cx, cy]
        pt1_ = [cx - s*self.x_scale*np.sin(ang),
                cy - s*self.y_scale*np.cos(ang)]
        pt2 = [pt1_[0] + 0.25*s*self.x_scale*np.sin(np.pi/2-ang),
               pt1_[1] - 0.25*s*self.y_scale*np.cos(np.pi/2-ang)]
        pt3 = [pt1_[0] - 0.25*s*self.x_scale*np.sin(np.pi/2-ang),
               pt1_[1] + 0.25*s*self.y_scale*np.cos(np.pi/2-ang)]
        vertex_list = []
        vertex_color = []
        for point in [pt1, pt2, pt3]:
            vertex_list += point
            vertex_color += color#[255, 0, 0]
        index = [x for x in range(3)]
        group = pyglet.graphics.Group()
        self.vehicle_batch.add_indexed(3, GL_POLYGON,
                                       group, index,
                                       ("v2f", vertex_list),
                                       ("c3B", vertex_color))


    def _add_vehicle_poly_circle(self, center, radius, color):
        if radius == 0:
            return
        cx, cy = center
        radius = radius * self.pxpm
        pxpm = self.pxpm*10
        vertex_list = []
        vertex_color = []
        for idx in range(pxpm):
            angle = np.radians(float(idx)/pxpm * 360.0)
            x = radius*self.x_scale*np.cos(angle) + cx
            y = radius*self.y_scale*np.sin(angle) + cy
            vertex_list += [x, y]
            vertex_color += color
        index = [x for x in range(pxpm)]
        group = pyglet.graphics.Group()
        self.vehicle_batch.add_indexed(pxpm, GL_LINE_LOOP,
                                       group, index,
                                       ("v2f", vertex_list),
                                       ("c3B", vertex_color))
