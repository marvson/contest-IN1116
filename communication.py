class Communication:
    def __init__(self):
        self.messages = []

    def pending(self, index):
        return [m for m in self.messages if m[1] != index]

    def say(self, something, index):
        self.messages.append((something, index))

    def clear(self, index):
        self.messages = [m for m in self.messages if m[1] != index]
