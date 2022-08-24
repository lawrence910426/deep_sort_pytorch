class Box:
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2


class Counter:
    def __init__(self, box: Box):
        # State descriptions
        # 0 = Neutral
        # 1 = Crossed over left border
        # 2 = Crossed over right border
        self.state = {}

    def getFlow(self):
        return {
            "Forward": 0,
            "Reverse": 0
        }

    def hover(self, a: Box, b: Box):
        # Ensures a.x1 < b.x1
        if a.x1 > b.x1:
            a, b = b, a
        
        if a.y1 < b.y2 and b.x1 < b.x2:
            return True
        if b.y1 < a.y2 and b.x1 < b.x2:
            return True

    def update(self, id, coord: Box):
        pass