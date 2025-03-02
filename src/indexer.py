# Minimal implementation to satisfy pyspellchecker’s import
class DictionaryIndex:
    def __init__(self, words=None):
        self.words = words if words is not None else []

    def build_index(self, words):
        # A simple approach: store the unique words in a list.
        self.words = list(set(words))
        return self.words

    def get_index(self):
        return self.words