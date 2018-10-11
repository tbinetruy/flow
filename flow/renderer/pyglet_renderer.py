import pyglet
from pyglet.gl import *
import numpy as np
import cv2, imutils
import os

class PygletRenderer():

    def __init__(self, kernel, save_frame=False, save_dir="./"):
        self.kernel = kernel
        self.save_frame = save_frame
        self.pxpm = 2 # Pixel per meter
        if self.save_frame:
            self.save_dir = save_dir

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
        self.width = width * self.pxpm
        self.x_shift = shift
        self.x_scale = scale

        polys_y = np.asarray(lane_polys_flat[1::2])
        height = int(polys_y.max() - polys_y.min())
        shift = polys_y.min() - 2
        scale = (height - 4) / height
        self.height = height * self.pxpm
        self.y_shift = shift
        self.y_scale = scale

        self.lane_colors = []
        for lane_poly in self.lane_polys:
            lane_poly[::2] = [(x-self.x_shift)*self.x_scale*self.pxpm
                              for x in lane_poly[::2]]
            lane_poly[1::2] = [(y-self.y_shift)*self.y_scale*self.pxpm
                               for y in lane_poly[1::2]]
            color = [c for _ in range(int(len(lane_poly)/2))
                     for c in [250, 250, 0]]
            self.lane_colors.append(color)

        self.window = pyglet.window.Window(width=self.width,
                                           height=self.height)
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        frame = frame.reshape(buffer.height, buffer.width, 4)
        self.frame = frame[::-1,:,0:3]
        print("Rendering with Pyglet with frame:", (self.width, self.height))


    def render(self, human_orientations, machine_orientations):
        glClearColor(0.125,0.125,0.125,1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self.lane_batch = pyglet.graphics.Batch()
        self.add_lane_polys()
        self.lane_batch.draw()
        self.vehicle_batch = pyglet.graphics.Batch()
        self.add_vehicle_polys(human_orientations, [0, 255, 0])#[0,139,139])
        self.add_vehicle_polys(machine_orientations, [255, 255, 255])#[255,20,147])
        self.vehicle_batch.draw()

        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        if self.save_frame:
            t = self.kernel.simulation.getCurrentTime()
            buffer.save("%s/frame%06d.png" % (self.save_dir, t))
        image_data = buffer.get_image_data()
        frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        frame = frame.reshape(buffer.height, buffer.width, 4)
        self.frame = frame[::-1,:,0:3].mean(axis=-1,keepdims=True)

        self.window.flip()
        return self.frame


    def get_sight(self, orientation, sight_radius):
        sight_radius = sight_radius * self.pxpm
        x, y, ang = orientation#; print(ang)
        x = (x-self.x_shift)*self.x_scale*self.pxpm
        y = (y-self.y_shift)*self.y_scale*self.pxpm
        x_med = x + sight_radius
        y_med = self.height - y + sight_radius # TODO: Check this!
        x_min = int(x_med - sight_radius)
        y_min = int(y_med - sight_radius)
        x_max = int(x_med + sight_radius)
        y_max = int(y_med + sight_radius)
        frame = np.squeeze(self.frame)
        padded_frame = np.pad(frame, int(sight_radius+1),
                              "constant", constant_values=32)
        fixed_sight = padded_frame[y_min:y_max, x_min:x_max].astype(np.uint8)
        height,width = fixed_sight.shape
        mask = np.zeros((height,width), np.uint8)
        cv2.circle(mask, (int(sight_radius), int(sight_radius)),
                   int(sight_radius), (255,255,255), thickness=-1)
        rotated_sight = cv2.cvtColor(fixed_sight, cv2.COLOR_GRAY2BGR)
        rotated_sight = cv2.bitwise_and(rotated_sight, rotated_sight, mask=mask)
        rotated_sight = imutils.rotate(rotated_sight, ang)
        return rotated_sight.mean(axis=-1,keepdims=True)


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


    def add_vehicle_polys(self, orientations, color):
        for orientation in orientations:
            x, y, ang = orientation
            x = (x-self.x_shift)*self.x_scale*self.pxpm
            y = (y-self.y_shift)*self.y_scale*self.pxpm
            self._add_vehicle_poly_triangle((x, y), ang, 5, color)
            #self._add_vehicle_poly_circle((x, y), 3, color)


    def _add_vehicle_poly_triangle(self, center, angle, size, color):
        cx, cy = center
        #print(angle)
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
        cx, cy = center
        r = radius*self.pxpm
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
        self.vehicle_batch.add_indexed(pxpm, GL_POLYGON,
                                       group, index,
                                       ("v2f", vertex_list),
                                       ("c3B", vertex_color))
