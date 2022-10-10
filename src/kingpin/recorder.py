"""
Record acceptance rates of Markov chain moves
=============================================
"""


class Record:
    """
    Record efficiency of Markov chain
    """

    def __init__(self, accept=0, reject=0):
        self.accept = accept
        self.reject = reject

    @property
    def efficiency(self):
        """
        :return: Efficiency of Markov chain moves
        """
        return self.accept / (self.accept + self.reject)

    def add(self, accept):
        """
        Add whether move was accepted or rejected
        """
        if accept:
            self.accept += 1
        else:
            self.reject += 1

    def __add__(self, other):
        """
        Combine records
        """
        return Record(self.accept + other.accept, self.reject + other.reject)


class Recorder(dict):
    """
    Record efficiency of several Markov chain moves
    """

    def __getitem__(self, key):
        """
        Fetch record or make new one
        """
        if key not in self:
            self[key] = Record()
        return super().__getitem__(key)

    def __str__(self):
        """
        Efficiencies of all Markov chain moves
        """
        eff_strs = [f"{k} = {v.efficiency:4.3f}" for k, v in self.items()]
        return ". ".join(eff_strs)

    def __add__(self, other):
        """
        Combine collections of records
        """
        result = Recorder()

        for key in self:
            if key in other:
                result[key] = self[key] + other[key]
            else:
                result[key] = self[key]

        for key in other:
            if key not in result:
                result[key] = other[key]

        return result
