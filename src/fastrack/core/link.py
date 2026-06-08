"""Link and Path: the data structures produced by frame-to-frame tracking.

A ``Link`` connects one filament in frame N to its partner in frame N+1; a
``Path`` is a chain of links forming a single filament's trajectory.  Moved
verbatim from the original ``motility.py``.
"""


class Link:
    def __init__(self):
        self.frame1_no = 0
        self.frame2_no = 0
        self.filament1_label = 0
        self.filament2_label = 0
        self.filament1_length = 0
        self.filament2_length = 0
        self.filament1_contour = []
        self.filament2_contour = []
        self.filament1_cm = []
        self.filament2_cm = []

        self.average_length = 0
        self.overlap_score = 0
        self.area_score = 0
        self.distance_score = 0

        self.fil_direction = 1
        self.mov_direction = 1

        self.dt = 0
        self.instant_velocity = 0

        self.forward_link = None
        self.reverse_link = None

        self.direct_link = False


class Path:
    def __init__(self):
        self.links = []
        self.first_frame_no = 0
        self.path_length = 0
        self.ave_fil_length = 0
        self.ave_velocity = 0
        self.std_velocity = 0
        self.max_velocity = 0
        self.min_velocity = 0

        self.stuck = False

