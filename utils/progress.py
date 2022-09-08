
class Progress:
    def __init__(self, l=0, r=100):
        self.left = l
        self.right = r
    
    def get_progress(self, progress_percent):
        progress = self.left + progress_percent / 100 * (self.right - self.left)
        progress = int(progress)
        return f"[Progress]---{progress}"

    def done(self):
        self.update(self.total)
        print()