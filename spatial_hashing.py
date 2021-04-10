from builtins import int, range


class SpatialHash:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.contents = {}

    def hash(self, point):
        return int(point[0]/self.cell_size), int(point[1]/self.cell_size), int(point[2]/self.cell_size)

    def insert_object_for_point(self, point, object):
        self.contents.setdefault(self.hash(point), []).append(object)

    def insert_object_for_box(self, box, object):
        # hash the minimum and maximum points
        min, max = self.hash(box.min), self.hash(box.max)
        # iterate over the rectangular region
        for i in range(min[0], max[0]+1):
            for j in range(min[1], max[1]+1):
                # append to each intersecting cell
                self.contents.setdefault((i, j), []).append(object)
