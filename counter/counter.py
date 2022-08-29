import numpy as np
from utils.shapes import Box, Line

class Counter:
    def __init__(self, fps, detector: Line):
        self.detector = detector
        self.fps = fps
        # Stores the inner product of last second between
        # detector and the vehicle as a queue. The queue
        # has length of fps.
        self.state = {}
        self.flow = {
            "Forward": 0,
            "Reverse": 0
        }

    def getFlow(self):
        return self.flow

    def hover(self, vehicle: Box):
        seg, rect = self.detector, vehicle
        inside = lambda x, y: rect.x1 <= x and x <= rect.x2 and rect.y1 <= y and y <= rect.y2
        if inside(seg.x1, seg.y1) and inside(seg.x2, seg.y2):
            return True
        
        formula = lambda x: (seg.y2 - seg.y1) / (seg.x2 - seg.x1) * (x - seg.x1) + seg.y1
        if rect.y1 <= formula(rect.x1) and formula(rect.x1) <= rect.y2 and \
           seg.x1 <= rect.x1 and rect.x1 <= seg.x2:
           return True
        if rect.y1 <= formula(rect.x2) and formula(rect.x2) <= rect.y2 and \
           seg.x1 <= rect.x2 and rect.x2 <= seg.x2:
           return True

    def update(self, id, vehicle: Box):
        if not (id in self.state):
            self.state[id] = { "Counted": False, "InnerProduct": [] }
        
        if self.state[id]["Counted"]:
            return

        if len(self.state[id]["InnerProduct"]) > self.fps:
            self.state[id]["InnerProduct"].pop(0)
        
        centroid = (vehicle.x1 + vehicle.x2) / 2, (vehicle.y1 + vehicle.y2) / 2
        midpoint = (self.detector.x1 + self.detector.x2) / 2, (self.detector.y1 + self.detector.y2) / 2
        vector = self.detector.x2 - self.detector.x1, self.detector.y2 - self.detector.y1
        inner_prod = np.dot(np.array(vector), np.array(centroid) - np.array(midpoint))
        self.state[id]["InnerProduct"].append(inner_prod)

        if self.hover(vehicle):
            self.state[id]["Counted"] = True
            positive = sum([1 if item > 0 else 0 for item in self.state[id]["InnerProduct"]])
            negative = self.fps - positive
            self.flow["Forward" if positive > negative else "Reverse"] += 1
