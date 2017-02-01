from math import pi, sin

# kSIMPLE_TRAIN = [(1, True), (2, True), (4, True), (5, True)]
# kSIMPLE_TRAIN = [(1, False), (2, True), (4, True), (5, True)]
# kSIMPLE_TRAIN = [(1, True), (2, False), (4, True), (5, True)]
# kSIMPLE_TRAIN = [(1, True), (2, True), (4, False), (5, True)]
kSIMPLE_TRAIN = [(1, True), (2, True), (4, True), (5, False)]
# kSIMPLE_TRAIN = [(1, False), (2, False), (4, True), (5, True)]
# kSIMPLE_TRAIN = [(1, False), (2, True), (4, False), (5, True)]
# kSIMPLE_TRAIN = [(1, False), (2, True), (4, True), (5, False)]

class SinClassifier:
    """
    A binary classifier that is parameterized a single float
    """

    def __init__(self, w):
        """
        Create a new classifier parameterized by w

        Args:
          w: The parameter w in the sin function (a real number)
        """
        assert isinstance(w, float)
        self.w = w

    def __call__(self, k):
        """
        Returns the raw output of the classifier.  The sign of this value is the
        final prediction.

        Args:
          k: The exponent in x = 2**(-k) (an integer)
        """
        return sin(self.w * 2 ** (-k))

    def classify(self, k):
        """

        Classifies an integer exponent based on whether the sign of \sin(w * 2^{-k})
        is >= 0.  If it is, the classifier returns True.  Otherwise, false.

        Args:
          k: The exponent in x = 2**(-k) (an integer)
        """
        assert isinstance(k, int), "Object to be classified must be an integer"

        if self(k) >= 0:
            return True
        else:
            return False

if __name__ == "__main__":

    for w in range(1,60):
        classifier = SinClassifier(w*pi)
        flag = True
        for kk, yy in kSIMPLE_TRAIN:
            if(yy != classifier.classify(kk)):
                flag = False
                break
        if(flag):
            print w
            break
