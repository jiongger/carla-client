import numpy as np
import carla

from .sensor_utils import _baseCustomSensor

__all__ = [
            'world_to_sensor', 
            'sensor_to_world', 
            'sensor_to_sensor', 
            'vehicle_bbox_to_sensor', 
            'vehicle_bboxes_to_sensor', 
            'vehicle_bbox_to_world', 
            'vehicle_bboxes_to_world',
          ]


def _complete_homography_matrix(matrix) -> np.array:
    if len(matrix.shape) == 1:
        matrix = np.expand_dims(matrix, axis=0)
    if matrix.shape[1] == 3:
        shift = np.ones((matrix.shape[0], 1))
        return np.concatenate([matrix, shift], axis=1)
    elif matrix.shape[1] == 4:
        return matrix
    else:
        raise NotImplementedError


def _create_bbox_points(vehicle) -> np.array:

    cords = np.zeros((8, 4))
    extent = vehicle.bounding_box.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return cords



def world_to_sensor(cords, sensor, homography=False) -> np.array:
    assert isinstance(cords, np.ndarray), type(cords)
    assert isinstance(sensor, (carla.Sensor, _baseCustomSensor)), type(sensor)

    cords = _complete_homography_matrix(cords)
    sensor_world_matrix = sensor.get_transform().get_matrix()
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, np.transpose(cords))
    if homography:
        return np.transpose(sensor_cords)
    else:
        return np.transpose(sensor_cords)[:, :3]



def sensor_to_world(cords, sensor, homography=False) -> np.array:
    assert isinstance(cords, np.ndarray), type(cords)
    assert isinstance(sensor, (carla.Sensor, _baseCustomSensor)), type(sensor)

    cords = _complete_homography_matrix(cords)
    sensor_world_matrix = sensor.get_transform().get_matrix()
    world_cords = np.dot(sensor_world_matrix, np.transpose(cords))
    if homography:
        return np.transpose(world_cords)
    else:
        return np.transpose(world_cords)[:, :3]



def sensor_to_sensor(cords, src, dst, homography=False) -> np.array:
    assert isinstance(cords, np.ndarray), type(cords)
    assert isinstance(src, (carla.Sensor, _baseCustomSensor)), type(src)
    assert isinstance(dst, (carla.Sensor, _baseCustomSensor)), type(dst)

    cords = _complete_homography_matrix(cords)
    src_world_cords = sensor_to_world(cords, src, True)
    dst_cords = world_to_sensor(src_world_cords, dst, True)
    if homography:
        return np.transpose(dst_cords)
    else:
        return np.transpose(dst_cords)[:, :3]



def vehicle_bbox_to_world(vehicle, homography=False) -> np.array:
    assert isinstance(vehicle, carla.Vehicle), type(vehicle)

    bbox_cords = _create_bbox_points(vehicle)
    bbox_transform = carla.Transform(vehicle.bounding_box.location)
    bbox_vehicle_matrix = bbox_transform.get_matrix()
    vehicle_world_matrix = vehicle.get_transform().get_matrix()
    bbox_world_matrix = np.dot(vehicle_world_matrix, bbox_vehicle_matrix)
    world_cords = np.dot(bbox_world_matrix, np.transpose(bbox_cords))
    if homography:
        return np.transpose(world_cords)
    else:
        return np.transpose(world_cords)[:, :3]



def vehicle_bboxes_to_world(vehicles, homography=False) -> list:
    assert isinstance(vehicles, list), type(vehicles)

    bboxes_world_cords = []
    for vehicle in vehicles:
        bbox = vehicle_bbox_to_world(vehicle, homography)
        bboxes_world_cords.append(bbox)
    return bboxes_world_cords



def vehicle_center_to_world(vehicle, homography=False) -> np.array:
    assert isinstance(vehicle, carla.Vehicle), type(vehicle)

    center = np.array([[0,0,0,1]], dtype=np.dtype("float32"))
    center_transform = carla.Transform(vehicle.bounding_box.location)
    center_vehicle_matrix = center_transform.get_matrix()
    vehicle_world_matrix = vehicle.get_transform().get_matrix()
    center_world_matrix = np.dot(vehicle_world_matrix, center_vehicle_matrix)
    center_world_cords = np.dot(center_world_matrix, np.transpose(center))
    if homography:
        return np.transpose(center_world_cords)
    else:
        return np.transpose(center_world_cords)[:, :3]



def vehicle_centers_to_world(vehicles, homography=False) -> list:
    assert isinstance(vehicles, list), type(vehicles)

    centers_world_cords = []
    for vehicle in vehicles:
        center = vehicle_center_to_world(vehicle, homography)
        centers_world_cords.append(center)
    return centers_world_cords



def vehicle_bbox_to_sensor(vehicle, sensor, homography=False) -> np.array:
    assert isinstance(vehicle, carla.Vehicle), type(vehicle)
    assert isinstance(sensor, (carla.Sensor, _baseCustomSensor)), type(sensor)

    bbox_world_cords = vehicle_bbox_to_world(vehicle, True)
    sensor_cords = world_to_sensor(bbox_world_cords, sensor, True)
    if homography:
        return np.transpose(sensor_cords)
    else:
        return np.transpose(sensor_cords)[:, :3]



def vehicle_bboxes_to_sensor(vehicles, sensor, homography=False) -> list:
    assert isinstance(vehicles, list), type(vehicles)
    assert isinstance(sensor, (carla.Sensor, _baseCustomSensor)), type(sensor)

    bboxes_sensor_cords = []
    for vehicle in vehicles:
        bbox = vehicle_bbox_to_sensor(vehicle, sensor, homography)
        bboxes_sensor_cords.append(bbox)
    return bboxes_sensor_cords



def vehicle_center_to_sensor(vehicle, sensor, homography=False) -> np.array:
    assert isinstance(vehicle, carla.Vehicle), type(vehicle)
    assert isinstance(sensor, (carla.Sensor, _baseCustomSensor)), type(sensor)

    center_world_cords = vehicle_bbox_to_world(vehicle, True)
    sensor_cords = world_to_sensor(center_world_cords, sensor, True)
    if homography:
        return np.transpose(sensor_cords)
    else:
        return np.transpose(sensor_cords)[:, :3]



def vehicle_centers_to_sensor(vehicles, sensor, homography=False) -> list:
    assert isinstance(vehicles, list), type(vehicles)
    assert isinstance(sensor, (carla.Sensor, _baseCustomSensor)), type(sensor)

    centers_sensor_cords = []
    for vehicle in vehicles:
        center = vehicle_center_to_sensor(vehicle, sensor, homography)
        centers_sensor_cords.append(center)
    return centers_sensor_cords
