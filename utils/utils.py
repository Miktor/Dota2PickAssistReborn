from sys import stdout
import numpy as np


def progress_bar(current, total):
    barLength = 20  # Modify this to change the length of the progress bar
    status = ""

    progress = float(current) / float(total)

    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1:.3f}% {2}".format("=" * block + " " * (barLength - block), progress * 100, status)
    stdout.write(text)
    stdout.flush()


class GenericMemory(object):
    def __init__(self, capacity, definitions):
        self.memory = np.empty(capacity, dtype=definitions)
        self.size = capacity
        self.i = 0
        self.filled = 0
        self.definitions = definitions

    def append(self, *args):
        self.memory[self.i] = args
        self.i = (self.i + 1) % self.size
        self.filled = min(self.filled + 1, self.size)

    def sample(self, size):
        indices = np.random.randint(0, self.filled, min(self.filled, size))
        batch = self.memory[indices]
        return (batch[col] for col, _, _ in self.definitions)

    def __len__(self):
        return self.filled

if __name__ == '__main__':
    pass
