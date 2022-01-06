import numpy as np
import cv2 as cv


BBOX_COLOR = (24, 64, 248)
PTS_COLOR = (248, 248, 248)

def _rotate_xyz(data) -> np.array:
    _v2c_m = np.array([
                        [ 0,  0,  1],
                        [ 1,  0,  0],
                        [ 0, -1,  0],
                      ])
    return np.dot(data, _v2c_m)

def project_points_to_camera(world_cords, camera):
    from .transform_utils import world_to_sensor
    calibration = camera.get_calibration_matrix()
    camera_cords = world_to_sensor(world_cords, camera)
    camera_cords_xyz = camera_cords[:, :3]
    # sensor coordinates to image coordinates
    camera_cords_rotated = _rotate_xyz(camera_cords_xyz)
    points = np.transpose(np.dot(calibration, np.transpose(camera_cords_rotated)))
    points_pixel = np.concatenate([points[:, 0:1] / points[:, 2:3], points[:, 1:2] / points[:, 2:3], points[:, 2:3]], axis=1)
    return points_pixel



def project_points_to_image(world_cords, camera, color=PTS_COLOR):
    '''Parameters:
            world_cords:    np.array(Nx3)   - projected points in world coordinate
            camera:         CustomCamera    - 
            data:           carla.Image     - carla image to be projected on
            color:          Tuple           - color of projected points
       Output:
            image:          np.array        - image of cv format (HxWx3, BGR)
    '''
    pixel_cords = project_points_to_camera(world_cords, camera)
    data = camera.data

    image = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    image = np.reshape(image, (data.height, data.width, 4))
    image = image[:, :, :3]
    point: np.array
    for point in pixel_cords:
        cv.circle(image, center=(round(point[0]), round(point[1])), radius=1, color=color)
    return image



def project_bboxes_to_image(world_bboxes, camera, color=BBOX_COLOR):
    '''Parameters:
            world_bboxes:   List[np.array]  - projected bboxes in world coordinate 
                                                 (bbox must be np.array with shape of (8, 4))
                                                 (and better created by module 'transform_utils')
            camera:         CustomCamera    - 
            data:           carla.Image     - carla image to be projected on
            color:          Tuple           - color of projected bboxes
       Output:
            image:          np.array        - image of cv format (HxWx3, BGR)
    '''
    data = camera.data

    image = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    image = np.reshape(image, (data.height, data.width, 4))
    image_bgr = np.ascontiguousarray(image[:, :, :3])
    for bbox in world_bboxes:
        bbox_image = project_points_to_camera(bbox, camera)
        points = []
        for i in range(8):
            points.append((round(bbox_image[i, 0]), round(bbox_image[i, 1])))

        if all(bbox_image[:, 2] > 0):
            cv.line(image_bgr, points[0], points[1], color=color)
            cv.line(image_bgr, points[1], points[2], color=color)
            cv.line(image_bgr, points[2], points[3], color=color)
            cv.line(image_bgr, points[3], points[0], color=color)

            cv.line(image_bgr, points[4], points[5], color=color)
            cv.line(image_bgr, points[5], points[6], color=color)
            cv.line(image_bgr, points[6], points[7], color=color)
            cv.line(image_bgr, points[7], points[4], color=color)

            cv.line(image_bgr, points[0], points[4], color=color)
            cv.line(image_bgr, points[1], points[5], color=color)
            cv.line(image_bgr, points[2], points[6], color=color)
            cv.line(image_bgr, points[3], points[7], color=color)
    return image_bgr