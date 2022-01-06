import sys
import os
import glob
try:
    sys.path.append(glob.glob('carla/dist/carla-*%d.%d-%s.egg' % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from clientlib.camera_utils import CustomCamera
from clientlib.lidar_utils import CustomLidar
from clientlib.kitti_utils import generate_kitti_label_file, generate_kitti_calib_file
from clientlib.carla_utils import save_snapshot
from clientlib.utils import make_dirs


class SynchronousClient:

    def __init__(self):
        self.client = None
        self.world = None
        self.manager = None
        self.number_of_cars = 40
        self.frames_per_second = 20

        self.ego = None
        self.spectator = None

        self.image_x = 1920
        self.image_y = 1080
        self.fov = 90
        self.sensor_tick = 0.1
        self.tick = -1

    def set_synchronous_mode(self, synchronous_mode):

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 1 / self.frames_per_second
        self.world.apply_settings(settings)
        self.manager.set_synchronous_mode(synchronous_mode)

    def setup_cars(self):

        car_bps = self.world.get_blueprint_library().filter('vehicle.*')
        car_bp_sample = []
        for _ in range(self.number_of_cars):
            car_bp_sample.append(random.choice(car_bps))
        location = random.sample(self.world.get_map().get_spawn_points(), self.number_of_cars)
        self.ego = self.world.spawn_actor(car_bps[0], location[0])
        self.ego.set_autopilot(True, self.manager.get_port())
        for i in range(1, self.number_of_cars):
            current_car = self.world.spawn_actor(car_bp_sample[i], location[i])
            current_car.set_autopilot(True, self.manager.get_port())

    def update_spectator(self, transform=[-5.5, 0.0, 2.8, -15.0, 0.0, 0.0]):
        ''' transform: = [x, y, z, pitch, yaw, roll] '''

        specatator_vehicle_transform = carla.Transform(location=carla.Location(*transform[0:3]), 
                                                       rotation=carla.Rotation(*transform[3:6]))
        specatator_vehicle_matrix = specatator_vehicle_transform.get_matrix()
        vehicle_world_matrix = self.ego.get_transform().get_matrix()
        specatator_world_matrix = np.dot(vehicle_world_matrix, specatator_vehicle_matrix)
        specatator_world = np.dot(specatator_world_matrix, np.transpose(np.array([[*transform[0:3], 1.0]], dtype=np.dtype("float32"))))
        specatator_rotation = self.ego.get_transform().rotation
        specatator_rotation.pitch = -15
        spectator_transform = carla.Transform(location=carla.Location(*specatator_world[0:3, 0]), 
                                              rotation=specatator_rotation)
        self.spectator.set_transform(spectator_transform)

    def setup_spectator(self):

        self.spectator = self.world.get_spectator()
        self.update_spectator()

    def setup_camera(self, transform, log_dir='dataset/kitti/image_2', suffix='', **options):
        ''' transform: = [x, y, z, pitch, yaw, roll] '''

        camera_location = carla.Location(*transform[0:3])
        camera_rotation = carla.Rotation(*transform[3:6])
        camera_transform = carla.Transform(location=camera_location, rotation=camera_rotation)
        return CustomCamera(self.world, camera_transform, self.ego, log_dir,  
                            suffix=suffix, 
                            with_bbox=True, **options)
    
    def setup_lidar(self, transform, log_dir='dataset/kitti/velodyne', **options):
        ''' transform: = [x, y, z, pitch, yaw, roll] '''

        lidar_location = carla.Location(*transform[0:3])
        lidar_rotation = carla.Rotation(*transform[3:6])
        lidar_transform = carla.Transform(location=lidar_location, rotation=lidar_rotation)
        return CustomLidar(self.world, lidar_transform, self.ego, log_dir, **options)

    def loop(self):
        camera_options = {
            'image_size_x': self.image_x,
            'image_size_y': self.image_y,
            'fov': self.fov,
            'sensor_tick': self.sensor_tick,
            'motion_blur_intensity': 0.0
        }
        lidar_options = {
            'range': 150,
            'channels': 64,
            'points_per_second': 2240000,
            'rotation_frequency': 20,
            'sensor_tick': self.sensor_tick,
            'upper_fov': 5.0,
            'lower_fov': -20.0
        }

        try:
            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()
            self.manager = self.client.get_trafficmanager(8000)
            self.set_synchronous_mode(True)

            self.setup_cars()
            self.setup_spectator()
            self.front = self.setup_camera([0.0,0.0,2.0,0.0,0.0,0.0], **camera_options)
            self.right = self.setup_camera([0.0,0.0,2.0,0.0,90.0,0.0], suffix='right', **camera_options)
            self.back = self.setup_camera([0.0,0.0,2.0,0.0,180.0,0.0], suffix='back', **camera_options)
            self.left = self.setup_camera([0.0,0.0,2.0,0.0,270.0,0.0], suffix='left', **camera_options)
            self.lidar = self.setup_lidar([0.0,0.0,2.0,0.0,0.0,0.0], **lidar_options)
            label_dir = make_dirs('dataset/kitti/label_2')
            calib_dir = make_dirs('dataset/kitti/calib')
            snap_dir = make_dirs('dataset/kitti/snaps')
            label_count = 0
            assert label_dir is not None 
            assert calib_dir is not None
            assert snap_dir is not None

            while True:
                self.tick += 1
                self.world.tick()
                self.update_spectator()
                print(self.front.retrive, 
                      self.right.retrive,
                      self.back.retrive,
                      self.left.retrive,
                      self.lidar.retrive)

                self.front.save_data()
                self.right.save_data()
                self.back.save_data()
                self.left.save_data()
                self.lidar.save_data()

                if self.tick % (self.sensor_tick // (1 /self.frames_per_second)) == 0:
                    label_count += 1
                    generate_kitti_label_file(label_dir / ("%06d.txt" %label_count), self.world, self.front)
                    generate_kitti_calib_file(calib_dir / ("%06d.txt" %label_count), self.front, lidar=self.lidar)
                    save_snapshot(snap_dir / ("%06d.txt" %label_count), self.world)

        finally:
            vehicles = self.world.get_actors().filter('vehicle.*')
            self.front.destroy()
            self.right.destroy()
            self.back.destroy()
            self.left.destroy()
            self.lidar.destroy()
            for vehicle in vehicles:
                vehicle.destroy()
            self.set_synchronous_mode(False)


def main():
    try:
        client = SynchronousClient()
        client.loop()
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
