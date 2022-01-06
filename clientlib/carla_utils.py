
import numpy as np

import carla

from .transform_utils import world_to_sensor, vehicle_bbox_to_world, vehicle_center_to_world
from .project_utils import project_points_to_camera
from .label_utils import _baseLabel, _baseLabelList

class CarlaLabel(_baseLabel):
    def __init__(self, x, y, z, h, w, l, ry):
        super(CarlaLabel, self).__init__(x, y, z, h, w, l, ry)

    def to_str(self):
        return "%.2f %.2f %.2f %.2f %.2f %.2f %.2f" %(self.x, self.y, self.z, self.h, self.w, self.l, self.ry)

    @staticmethod
    def fromlist(list):
        assert len(list) == 7, len(list)
        label = CarlaLabel(*list)
        return label


class CarlaLabelList(_baseLabelList):
    def __init__(self):
        super(CarlaLabelList, self).__init__()

    @staticmethod
    def fromarray(array):
        assert isinstance(array, np.ndarray), type(array)
        _labellist = CarlaLabelList()
        for row in array:
            _labellist = _labellist + CarlaLabel.fromlist(row)
        return _labellist



def get_vehicles(world) -> list:
    return list(world.get_actors().filter('vehicle.*'))

def get_other_vehicles(world, vehicle) -> list:
    vehicles = get_vehicles(world)
    print(len(vehicles), type(vehicles))
    other_vehicles = []
    for v in vehicles:
        if v.id != vehicle.id:
            other_vehicles.append(v)
    return other_vehicles

def get_visible_vehicles(world, sensor) -> list:
    vehicles = get_vehicles(world)
    ego = sensor.parent
    visible_vehicles = []
    for vehicle in vehicles:
        if vehicle.id != ego.id and get_visible_flag(world, sensor, vehicle):
            visible_vehicles.append(vehicle)
    return visible_vehicles

def get_visible_flag(world, sensor, vehicle) -> bool:
    departure = sensor.get_location()
    destinations = vehicle_bbox_to_world(vehicle)
    labels = []
    for dst in destinations:
        sem = world.cast_ray(departure, carla.Location(*dst))
        labels.append(len(sem) == 0)
    return any(labels)



def get_labels(world, sensor, visible_only = False) -> CarlaLabelList:
    ego = sensor.parent
    ego_heading = ego.get_transform().rotation.yaw
    if visible_only:
        other_vehicles = get_visible_vehicles(world, sensor)
    else:
        other_vehicles = get_other_vehicles(world, ego)
    if len(other_vehicles) == 0:
        return CarlaLabelList()

    other_vehicles_location = []
    other_vehicles_hwl = []
    other_vehicles_heading = []
    for vehicle in other_vehicles:
        location = vehicle.get_location()
        other_vehicles_location.append(np.array([[location.x, location.y, location.z]]))
        extent = vehicle.bounding_box.extent
        other_vehicles_hwl.append(np.array([[extent.z * 2, extent.y * 2, extent.x * 2]]))
        heading = vehicle.get_transform().rotation.yaw
        other_vehicles_heading.append(np.array([[np.deg2rad(heading - ego_heading)]]))
    other_vehicles_location = np.concatenate(other_vehicles_location, axis=0)
    other_vehicles_hwl = np.concatenate(other_vehicles_hwl, axis=0)
    other_vehicles_heading = np.concatenate(other_vehicles_heading, axis=0)

    other_vehicles_location = world_to_sensor(other_vehicles_location, sensor)
    labels_xyzhwlry = np.concatenate([other_vehicles_location, other_vehicles_hwl, other_vehicles_heading], axis=1)
    labels = CarlaLabelList.fromarray(labels_xyzhwlry)
    return labels


def get_bboxes(world, camera, visible_only = False) -> np.array:
    ego = camera.parent
    if visible_only:
        other_vehicles = get_visible_vehicles(world, camera)
    else:
        other_vehicles = get_other_vehicles(world, ego)
    if len(other_vehicles) == 0:
        return np.array([[]])

    bbox = []
    for vehicle in other_vehicles:
        bbox3d = vehicle_bbox_to_world(vehicle)
        bbox2d = project_points_to_camera(bbox3d, camera)
        if all(bbox2d[:, 2] > 0):
            corner2d = np.array([[
                                    np.min(bbox2d[:, 0]), np.min(bbox2d[:, 1]),
                                    np.max(bbox2d[:, 0]), np.max(bbox2d[:, 1])
                                ]], dtype=np.dtype("float32"))
        else:
            corner2d = np.array([[
                                    -1000.0, -1000.0, -1000.0, -1000.0
                                ]], dtype=np.dtype("float32"))
        bbox.append(corner2d)
    bbox = np.concatenate(bbox, axis=0)
    if len(bbox.shape) == 1:
        return np.expand_dims(bbox, axis=0)
    else:
        return bbox

def get_vehicle_fine_classification(vehicle) -> str:
    return "Car"

def save_snapshot(path, world):
    vehicles = get_vehicles(world)
    _snapshot = open(path, 'w')
    print('saving snapshot to %s' %path)
    for vehicle in vehicles:
        location = vehicle.get_location()
        rotation = vehicle.get_transform().rotation
        extent = vehicle.bounding_box.extent
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        print('%d %s %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' %(
            vehicle.id, vehicle.type_id, 
            location.x, location.y, location.z, 
            extent.z*2, extent.y*2, extent.x*2, 
            rotation.pitch, rotation.yaw, rotation.roll,
            velocity.x, velocity.y, velocity.z,
            acceleration.x, acceleration.y, acceleration.z
        ), file=_snapshot)
    _snapshot.close()
