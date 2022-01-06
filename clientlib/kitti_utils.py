import numpy as np

from .label_utils import _baseLabel, _baseLabelList
__all__ = [
    'KittiLabel',
    'KittiLabelList',
    'save_as_kitti_velo',
    'get_kitti_label',
    'generate_kitti_label_file',
]

class KittiLabel(_baseLabel):
    __map_id_to_str = {
        1: 'Car',
        2: 'Van',
        3: 'Truck',
        4: 'Pedestrian',
        5: 'Person_sitting',
        6: 'Cyclist',
        7: 'Tram',
        8: 'Misc',
        0: 'DontCare'
    }

    def __init__(self, type, truncated, occluded, alpha, left, top, right, bottom, h, w, l, x, y, z, ry):
        super(KittiLabel, self).__init__(x, y, z, h, w, l, ry)
        if isinstance(type, str):
            self.type = type
        else:
            self.type = KittiLabel.type_id_to_str(type)
        self.truncated = truncated
        self.occluded = int(occluded)
        self.alpha = alpha
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.dimensions = np.array([h, w, l])
        self.location = np.array([x, y, z])
        self.bbox = np.array([[left, top], [right, bottom]])

    def to_str(self):
        return "%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" \
                    %(str(self.type), self.truncated, int(self.occluded), self.alpha,
                      self.left, self.top, self.right, self.bottom,
                      self.h, self.w, self.l,
                      self.x, self.y, self.z,
                      self.ry)

    @staticmethod
    def fromlist(list):
        assert len(list) == 15, len(list)
        return KittiLabel(*list)

    @staticmethod
    def type_id_to_str(id):
        return KittiLabel.__map_id_to_str[int(id)]
    @staticmethod
    def type_str_to_id(s):
        for key, value in KittiLabel.__map_id_to_str.items():
            if value == s:
                return key
        raise ValueError

    @staticmethod
    def fromannotation(annotation):
        assert isinstance(annotation, str)
        _list = annotation.split(' ')
        assert len(_list) == 15, len(_list)
        for i in range(1, 15):
            _list[i] = float(_list[i])
        _list[2] = int(_list[2])
        return KittiLabel(*list)

class KittiLabelList(_baseLabelList):
    def __init__(self):
        super(KittiLabelList, self).__init__()

    @staticmethod
    def fromarray(array):
        assert isinstance(array, np.ndarray), type(array)
        _labellist = KittiLabelList()
        for row in array:
            _labellist += KittiLabel.fromlist(row)
        return _labellist

    @staticmethod
    def fromannotations(annotations):
        assert isinstance(annotations, list), type(annotations)
        _labellist = KittiLabelList()
        for line in annotations:
            _labellist += KittiLabel.fromannotation(line)
        return _labellist



def _carla_lidar_to_kitti_lidar(data, use_intensity=True) -> np.ndarray:
    _transform = np.identity(4)
    _transform[1, 1] = -1
    kitti_xyzi = np.dot(data, _transform)
    if not use_intensity:
        i = np.ones((data.shape[0], 1))
        kitti_xyzi[:, 4] = i
    return np.ascontiguousarray(kitti_xyzi, dtype=np.dtype("float32"))

def save_as_kitti_velo(data, name, use_intensity=True) -> None:
    points = np.frombuffer(data.raw_data, dtype=np.dtype("float32"))
    points = np.reshape(points, (-1, 4))
    kitti_points = _carla_lidar_to_kitti_lidar(points, use_intensity)
    kitti_points.tofile(str(name))


def get_truncated(bbox, height, width) -> float:
    left, top, right, bottom = bbox
    if left == right or top == bottom:
        return 1.0
    _left = min(max(0, left), width-1)
    _top = min(max(0, top), height-1)
    _right = max(min(width-1, right), 0)
    _bottom = max(min(height-1, bottom), 0)
    return min(1.0 - (_right - _left) / (right - left) * (_bottom - _top) / (bottom - top), 1.0)
def get_alpha(x, z, ry) -> float:
    theta = -np.arcsin(x/np.sqrt(x**2+z**2))
    return ry - theta
def get_occluded(**kwargs) -> int:
    return 3

def tune_bbox(bbox, height, width) -> np.ndarray:
    left, top, right, bottom = bbox
    valid_x = (0 <= left <= width-1) or (0 <= right <= width-1)
    valid_y = (0 <= top <= height-1) or (0 <= bottom <= height-1)
    if valid_x and valid_y:
        return np.array([max(0.0, left), max(0.0, top), min(right, width-1), min(bottom, height-1)], 
                        dtype=np.dtype("float32"))
    else:
        return np.array([-1000.0, -1000.0, -1000.0, -1000.0], dtype=np.dtype("float32"))


def _rotate_xyz(data, velo_to_cam = True) -> np.ndarray:
    if len(data.shape) == 1:
        data = np.reshape(data, (1, -1))
    if velo_to_cam:
        _v2c_m = np.array([
                            [ 0,  0,  1],
                            [ 1,  0,  0],
                            [ 0, -1,  0],
                          ])
        return np.dot(data, _v2c_m)
    else:
        _c2v_m = np.array([
                            [ 0,  1,  0],
                            [ 0,  0, -1],
                            [ 1,  0,  0],
                          ])
        return np.dot(data, _c2v_m)


def get_kitti_label(world, camera) -> KittiLabelList:
    from .carla_utils import get_bboxes, get_labels, get_visible_vehicles, get_vehicle_fine_classification
    labels = get_labels(world, camera, visible_only=True)
    bboxes = get_bboxes(world, camera, visible_only=True)
    vehicles = get_visible_vehicles(world, camera)
    kittilabels = []
    for (vehicle, label, bbox) in zip(vehicles, labels, bboxes):
        v_type = get_vehicle_fine_classification(vehicle)
        tru = get_truncated(bbox, camera.image_height, camera.image_width)
        occ = get_occluded()
        alp = get_alpha(label.y, label.x, label.ry)
        kitti_label = [[
            KittiLabel.type_str_to_id(v_type), tru, occ, alp, 
            *tune_bbox(bbox, camera.image_height, camera.image_width), 
            *(label.get_extent()), 
            *(_rotate_xyz(label.get_location(), True)[0]), 
            label.get_heading() - np.radians(90)
        ]]
        kittilabels.append(np.array(kitti_label, dtype=np.dtype("float32")))
    if len(kittilabels) == 0:
        return KittiLabelList()
    kittilabels = np.concatenate(kittilabels, axis=0)
    return KittiLabelList.fromarray(kittilabels)

def generate_kitti_label_file(path, world, camera) -> None:
    kittilabels = get_kitti_label(world, camera)
    _label = open(path, 'w')
    print('saving label to %s' %path)
    for kittilabel in kittilabels:
        print(kittilabel.to_str(), file=_label)
    _label.close()


def get_kitti_calib(*cameras, lidar=None, imu=None) -> dict:
    calibration = {}
    P0_offset = cameras[0].get_offset()
    P0_bi = -P0_offset.location.y
    for i,camera in enumerate(cameras):
        index = "P%d" %i
        matrix = camera.get_calibration_matrix()
        Pi_offset = camera.get_offset()
        Pi_bi = -Pi_offset.location.y
        Pi_fu = matrix[0,0]
        Pi_tr = np.array([[Pi_fu * (Pi_bi - P0_bi)], [0], [0]], dtype=np.dtype("float32"))
        Pi_m = np.hstack([matrix, Pi_tr])
        calibration[index] = Pi_m.flatten()
    calibration['R0_rect'] = np.identity(3, dtype=np.dtype("float32")).flatten()
    if lidar is not None:
        velo_world_matrix = lidar.get_transform().get_matrix()
        cam_world_matrix = cameras[0].get_transform().get_matrix()
        world_cam_matrix = np.linalg.inv(cam_world_matrix)
        velo_cam_matrix = np.dot(velo_world_matrix, world_cam_matrix)
        tr_velo_cam = np.array([
                                    [ 0, -1,  0,  0],
                                    [ 0,  0, -1,  0],
                                    [ 1,  0,  0,  0],
                                    [ 0,  0,  0,  1]
                               ], dtype=np.dtype("float32"))
        Tr_velo_to_cam = np.dot(tr_velo_cam, velo_cam_matrix)
        calibration['Tr_velo_to_cam'] = Tr_velo_to_cam[:3].flatten()
        if imu is not None:
            imu_world_matrix = imu.get_transform().get_matrix()
            world_velo_matrix = np.linalg.inv(velo_world_matrix)
            imu_velo_matrix = np.dot(imu_world_matrix, world_velo_matrix)
            calibration['Tr_imu_to_velo'] = imu_velo_matrix[:3].flatten()
    return calibration

def generate_kitti_calib_file(path, *cameras, lidar=None, imu=None) -> None:
    calibration = get_kitti_calib(*cameras, lidar=lidar, imu=imu)
    _calib = open(path, 'w')
    print('saving calib to %s' %path)
    for name, matrix in calibration.items():
        print('%s:' %name, file=_calib, end=' ')
        for num in matrix[:-1]:
            print('%.12e' %num, file=_calib, end=' ')
        print('%.12e' %matrix[-1], file=_calib)
    _calib.close()
