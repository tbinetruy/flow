import pyglet
from pyglet.gl import *
import numpy as np
import traci.constants as tc
import os

class PygletRenderer():

    def __init__(self, kernel, save_frame=False, save_dir=None):
        self.kernel = kernel
        self.batch = pyglet.graphics.Batch()
        self.save_frame = save_frame
        self.pxpm = 3 # Pixel per meter
        if self.save_frame:
            self.save_dir = save_dir

        self.lane_polys = []
        for lane_id in self.kernel.lane.getIDList():
            _lane_poly = self.kernel.lane.getShape(lane_id)
            lane_poly = [i for pt in _lane_poly for i in pt ]
            self.lane_polys += lane_poly

        polys_x = np.asarray(self.lane_polys[::2])
        width = int(polys_x.max() - polys_x.min())
        shift = polys_x.min() - 2
        scale = (width - 4) / width
        self.width = width * self.pxpm
        self.lane_polys[::2] = [(x-shift)*scale for x in polys_x]
        self.x_shift = shift
        self.x_scale = scale

        polys_y = np.asarray(self.lane_polys[1::2])
        height = int(polys_y.max() - polys_y.min())
        shift = polys_y.min() - 2
        scale = (height - 4) / height
        self.height = height * self.pxpm
        self.lane_polys[1::2] = [(y-shift)*scale for y in polys_y]
        self.y_shift = shift
        self.y_scale = scale

        self.window = pyglet.window.Window(width=self.width,
                                           height=self.height)
        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
        frame = frame.reshape(buffer.height, buffer.width, 4)
        self.frame = frame[::-1,:,0:3]#; print(self.frame.shape)
        print("Frame dimension:", (self.width, self.height))


    def render(self):
        try:
            glClearColor(0,0,0,1)
            self.window.clear()
            self.window.switch_to()
            self.window.dispatch_events()

            self.batch = pyglet.graphics.Batch()
            self.add_lane_polys()
            self.add_vehicle_polys()
            self.batch.draw()
            #print(self.kernel.simulation.getCurrentTime())

            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            if self.save_frame:
                t = self.kernel.simulation.getCurrentTime()
                buffer.save("%s/frame%06d.png" % (self.save_dir, t))
            image_data = buffer.get_image_data()
            frame = np.fromstring(image_data.data, dtype=np.uint8, sep='')
            frame = frame.reshape(buffer.height, buffer.width, 4)
            self.frame = frame[::-1,:,0:3]

            self.window.flip()
            return self.frame
        except:
            self.close()
            return self.frame


    def close(self):
        self.window.close()


    def add_lane_polys(self):
        for lane_id in self.kernel.lane.getIDList():
            _polys = self.kernel.lane.getShape(lane_id)
            polys = [i for pt in _polys for i in pt]
            polys[::2] = [(x-self.x_shift)*self.x_scale*self.pxpm
                          for x in polys[::2]]
            polys[1::2] = [(y-self.y_shift)*self.y_scale*self.pxpm
                           for y in polys[1::2]]
            colors = [c for _ in _polys for c in [255, 255, 0]]
            self._add_lane_poly(polys, colors)


    def _add_lane_poly(self, polys, colors):
        num = int(len(polys)/2)
        index = [x for x in range(num)]
        group = pyglet.graphics.Group()
        self.batch.add_indexed(num, GL_LINE_STRIP, group, index,
                               ("v2f", polys), ("c3B", colors))


    def add_vehicle_polys(self):
        for veh_id in self.kernel.vehicle.getIDList():
            x, y = self.kernel.vehicle.getPosition(veh_id)
            x = (x-self.x_shift)*self.x_scale*self.pxpm
            y = (y-self.y_shift)*self.y_scale*self.pxpm
            ang = self.kernel.vehicle.getAngle(veh_id)
            #c = self.kernel.vehicle.getColor(veh_id)
            if "rl" in veh_id:
                c = [255,20,147] # Deeppink
            else:
                c = [0,139,139] # Darkcyan
            sc = self.kernel.vehicle.getShapeClass(veh_id)
            self._add_vehicle_poly_triangle((x, y), ang, 5, c)
            #self._add_vehicle_poly_circle((x, y), 3, c)


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
        self.batch.add_indexed(3, GL_POLYGON,
                               group, index,
                               ("v2f", vertex_list),
                               ("c3B", vertex_color))


    def _add_vehicle_poly_circle(self, center, radius, color):
        cx, cy = center
        r = radius*self.pxpm
        pxpm = self.pxpm*4
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
        self.batch.add_indexed(pxpm, GL_POLYGON,
                               group, index,
                               ("v2f", vertex_list),
                               ("c3B", vertex_color))
