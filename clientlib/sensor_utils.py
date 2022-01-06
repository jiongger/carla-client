import weakref

class _baseCustomSensor:
    def __init__(self, world, attached, transform, sensor_type, **params):
        self.world = world
        self.parent = attached
        self.offset = transform
        self.sensor = self.__init_sensor(transform, attached, sensor_type, **params)
        self.attributes = self.sensor.attributes
        self.id = self.sensor.id
        self.is_alive = self.sensor.is_alive
        self.semantic_tags = self.sensor.semantic_tags
        self.data = None
        self.retrive = True
        self.tick = 0

    def __init_sensor(self, transform, attached, sensor_type, **params):
        sensor_bp = self.world.get_blueprint_library().find(sensor_type)
        for key in params:
            sensor_bp.set_attribute(key, str(params[key]))
        sensor = self.world.spawn_actor(sensor_bp, transform, attach_to=attached)
        weak_self = weakref.ref(self)
        sensor.listen(lambda data: weak_self().__set_data(weak_self, data))
        return sensor

    @staticmethod
    def __set_data(weak_self, data):
        self = weak_self()
        if self.retrive:
            self.tick += 1
            self.data = data
            self.retrive = False

    def save_data(self):
        raise NotImplementedError

    def get_transform(self):
        return self.sensor.get_transform()
    def get_location(self):
        return self.sensor.get_location()
    def get_world(self):
        return self.world
    def destroy(self):
        self.sensor.stop()
        return self.sensor.destroy()
    def get_offset(self):
        return self.offset