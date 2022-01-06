import numpy as np

class _baseLabel(object):
    def __init__(self, x=0, y=0, z=0, h=0, w=0, l=0, ry=0):
        self.x = x
        self.y = y
        self.z = z
        self.h = h
        self.w = w
        self.l = l
        self.ry = ry

    def get_location(self):
        return np.array([self.x, self.y, self.z])
    def get_extent(self):
        return np.array([self.h, self.w, self.l])
    def get_heading(self):
        return self.ry

    def __str__(self):
        return str(self.__dict__)
    __repr__ = __str__


class _baseLabelList(object):
    def __init__(self):
        self.labels = []
        self.__iterater = 0
    def __eq__(self, o):
        return self.labels == o.labels if type(self) == type(o) else TypeError
    def __add__(self, x):
        if isinstance(x, _baseLabel):
            self.labels.append(x)
            return self
        elif isinstance(x, _baseLabelList):
            self.labels.extend(x.labels)
            return self
        else:
            raise NotImplementedError
    def __getitem__(self, index):
        assert isinstance(index, (int, np.int)), type(index)
        return self.labels[index]
    def __iter__(self):
        return self
    def __next__(self):
        if self.__iterater < len(self.labels):
            self.__iterater += 1
            return self.labels[self.__iterater - 1]
        else:
            raise StopIteration
    def __len__(self):
        return len(self.labels)
    def __str__(self):
        return str(self.labels)
    __repr__ = __str__
