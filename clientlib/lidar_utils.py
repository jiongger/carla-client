import numpy as np
import carla

from .sensor_utils import _baseCustomSensor


class CustomLidar(_baseCustomSensor):

    def __init__(self, world: carla.World, 
                       transform: carla.Transform, 
                       attached: carla.Actor,
                       log_to: str,
                       format: str = 'kitti',
                       use_intensity: bool = True,
                       **lidar_options):
        from .utils import make_dirs
        self.tick = 0
        self.log_to = make_dirs(log_to)
        super(CustomLidar, self).__init__(world, attached, transform, 'sensor.lidar.ray_cast', **lidar_options)
        self.format = format.lower()
        self.use_intensity = use_intensity

    def save_data(self):
        from .kitti_utils import save_as_kitti_velo
        if not self.retrive:
            _velo = self.log_to / ("%06d.ply" %self.tick)
            self.data.save_to_disk(str(_velo))
            if self.format == 'kitti':
                _velo = self.log_to / ("%06d.bin" %(self.tick))
                print('saving velodyne to %s' %_velo)
                save_as_kitti_velo(self.data, _velo, self.use_intensity)
            else:
                raise NotImplementedError
            self.retrive = True
        return self.retrive