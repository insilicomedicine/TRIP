class TrainStats():
    def __init__(self):
        self.stats = dict()

    def update(self, delta):
        for key in delta.keys():
            if key in self.stats.keys():
                self.stats[key].append(delta[key])
            else:
                self.stats[key] = [delta[key]]

    def reset(self):
        for key in self.stats.keys():
            self.stats[key] = []

    def print(self):
        for key in self.stats.keys():
            print(str(key) + ": {:4.4};".format(
                sum(self.stats[key]) / len(self.stats[key])), end='')

        print()