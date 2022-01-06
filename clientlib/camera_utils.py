import cv2 as cv
import numpy as np

import carla

from .sensor_utils import _baseCustomSensor


class CustomCamera(_baseCustomSensor):

    def __init__(self, world: carla.World, 
                       transform: carla.Transform, 
                       attached: carla.Actor,
                       log_to: str,
                       prefix: str = '',
                       suffix: str = '',
                       with_bbox: bool = False,
                       **camera_options):
        from .utils import make_dirs
        self.tick = 0
        self.log_to = make_dirs(log_to)
        super(CustomCamera, self).__init__(world, attached, transform, 'sensor.camera.rgb', **camera_options)
        self.calibration = self.__init_calibration_matrix()
        self.with_bbox = with_bbox
        self.prefix = prefix
        self.suffix = suffix

    def save_data(self):
        from .project_utils import project_bboxes_to_image
        from .carla_utils import get_visible_vehicles
        from .transform_utils import vehicle_bboxes_to_world
        if not self.retrive:
            _image = self.log_to / ("%s%06d%s.png" %(self.prefix, self.tick, self.suffix))
            print('saving image to %s' %_image)
            self.data.save_to_disk(str(_image))
            if self.with_bbox:
                _image_bbox = str(self.log_to / ("%s%06d%s_bbox.png" %(self.prefix, self.tick, self.suffix)))
                print('saving image with bounding box to %s' %_image_bbox)
                vehicle_bboxes_world = vehicle_bboxes_to_world(get_visible_vehicles(self.world, self))
                image = project_bboxes_to_image(vehicle_bboxes_world, self)
                cv.imwrite(_image_bbox, image)
            self.retrive = True
        return self.retrive


    def __init_calibration_matrix(self):
        image_x = float(self.sensor.attributes['image_size_x'])
        image_y = float(self.sensor.attributes['image_size_y'])
        fov = float(self.sensor.attributes['fov'])
        calibration = np.identity(3)
        calibration[0, 2] = image_x / 2.0
        calibration[1, 2] = image_y / 2.0
        calibration[0, 0] = calibration[1, 1] = image_x / (2.0 * np.tan(fov * np.pi / 360.0))
        self.image_width = image_x
        self.image_height = image_y
        self.fov = fov
        return calibration
    
    def get_calibration_matrix(self):
        return self.calibration
