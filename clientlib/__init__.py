import glob
import sys
import os
print('importing carla library from ../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
                    sys.version_info.major,
                    sys.version_info.minor,
                    'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
