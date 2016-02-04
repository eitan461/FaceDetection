import numpy as np
import funcs
from scipy import ndimage
from MySvm import MySvm
from AbstractClassifier import AbstractClassfier

class _Feature:
    def __init__(self, trans, plus, minus):
        self.plus = plus
        self.minus = minus
        self.trans = trans

    def calc(self, window_start):
        i, j = window_start
        result = 0
        for (x, y) in self.plus:
            r = ((i+x[0], i+x[1]), (j+y[0], j+y[1]))
            result += self.trans.intimg.get_sum(r)
        for (x, y) in self.minus:
            r = ((i+x[0], i+x[1]), (j+y[0], j+y[1]))
            result -= self.trans.intimg.get_sum(r)
        return result


class _Transform:
    def __init__(self, func):
        self.func = func
        self.mat = None
        self.intimg = None

    def change_image(self, img):
        self.mat = self.func(img).astype('float64')
        self.intimg = funcs.IntImg(self.mat)


class _SubClassifier(AbstractClassfier):
    def from_list(self, mat):
        self.svm.from_list(mat)

    def classify_vec(self, rect, axis=-1):
        return self.svm.classify_vec(self.calc_features(rect), axis)

    def to_list(self):
        return self.svm.to_list()

    def __init__(self, features, false_positive_loss=1, false_negative_loss=1):
        super().__init__()
        self.svm = MySvm()
        self._features = features
        self.svm.false_negative_loss = false_negative_loss
        self.svm.false_positive_loss = false_positive_loss
        self._row_len = len(features) + 1
        self.examples = np.empty((0, self._row_len))

    def add_current_image(self, y):
        row = np.empty(self._row_len)
        row[-1] = y
        row[:-1] = self.calc_features()
        self.examples = np.vstack((self.examples, row))

    def calc_features(self, rect=None):
        start = (0, 0) if rect is None else (rect[0][0], rect[1][0])
        return np.array([f.calc(start) for f in self._features])

    def learn(self):
        self.svm.learn(np.copy(self.examples), c_arr=2**np.arange(-5, 15, 2.0),
                       epoch=15, cross_validation_times=5, learning_part=0.7)

    def classify(self, rect):
        return self.svm.classify(self.calc_features(rect))

    def valuefy(self, rect):
        return self.svm.valuefy(self.calc_features(rect))


class MyViolaClassifier(AbstractClassfier):
    def __init__(self):
        super().__init__()
        self.rejecters = []
        self.sorted_rejecters = []
        self.plus_minus_rects = []
        self.transform_lists = []
        self.img = None
        self.add_basic_transforms()
        self.add_basic_features()
        self.add_basic_classifiers()
        self.num_examples = 0

    def change_image(self, img):
        self.img = img
        self.apply_image()

    def apply_image(self):
        for t_list in self.transform_lists:
            for t in t_list:
                t.change_image(self.img)

    def add_basic_transforms(self):
        rgb = [_Transform(lambda im, k=i: im[..., k]) for i in range(3)]
        laplace = [_Transform(lambda im, k=i: ndimage.filters.laplace(im)[..., k]) for i in range(3)]
        grad = [_Transform(lambda im, k=i: np.arctan2(ndimage.sobel(im[..., k], 0), ndimage.sobel(im[..., k], 1)))
                for i in range(3)] + [_Transform(
                lambda im, k=i: np.hypot(ndimage.sobel(im[..., k], 0), ndimage.sobel(im[..., k], 1))) for i in range(3)]
        # svm = Svm()
        # svm.load('svm_skin.pkl')
        # skin = [_Transform(lambda im: svm.classify_vec(im))]
        self.transform_lists += [rgb, laplace, grad]

    def add_basic_features(self):
        def add(l): self.plus_minus_rects.append(l)
        add([([((0, 69), (0, 69))], [])])  # whole image
        add([([((53, 67), (18, 52)), ((53, 67), (18, 52))], [((50, 69), (10, 60))])])  # mouth vs around mouth
        add([([((36, 52), (27, 43))], [((36, 52), (10, 26)), ((36, 52), (44, 60))]),  # nose vs sides of nose
             ([((20, 35), (10, 60))], [((36, 52), (10, 60))])])  # eyes vs nose
        add([([((20, 35), (10, 60))], [((4, 19), (10, 60))])])  # eyes vs forehead
        add([([((20, 35), (10, 30)), ((20, 35), (40, 60))], []),  # eyes only
             ([((36, 52), (22, 48))], [])])  # nose only
        add([([((53, 67), (18, 52))], [])])  # mouth only
        add([([((0, 69), (0, 34))], [((0, 69), (50, 69))])])  # horizontal symmetric
        add([([((0, 69), (0, 69))], [((0, 69), (4, 65)), ((0, 69), (4, 65))])])  # horizontal edges

    def add_basic_classifiers(self):
        for t_list in self.transform_lists:
            for f_list in self.plus_minus_rects:
                features = []
                for t in t_list:
                    for f in f_list:
                        features.append(_Feature(t, f[0], f[1]))
                self.rejecters.append(_SubClassifier(features, false_negative_loss=20))

    def add_examples(self, imgs, y):
        for img in imgs:
            self.change_image(img)
            self.num_examples += 1
            for classifier in self.rejecters:
                classifier.add_current_image(y)

    def learn(self):
        for classifier in self.rejecters:
            classifier.learn()
            classifier.svm._w /= np.linalg.norm(classifier.svm._w)
        self.sorted_rejecters = sorted(self.rejecters, key=lambda c: c.svm.error)

    def to_list(self):
        return [c.to_list() for c in self.rejecters]

    def from_list(self, list_):
        for c, l in zip(self.rejecters, list_):
            if not isinstance(l, list):
                raise ValueError('MyViolaClassifier.from_list: list_ has to be list of lists')
            c.from_list(l)
        self.sorted_rejecters = sorted(self.rejecters, key=lambda c: c.svm.error)

    def classify(self, rect):
        return self.valuefy(rect) > 0

    def valuefy(self, rect):
        failed = 0
        sum_ = 0
        for classifier in self.sorted_rejecters:
            cur = classifier.valuefy(rect)
            if cur < 0:
                failed += 1 / (1 + classifier.svm.error)
                if failed > 5.5:
                    return -1
            else:
                sum_ += cur / (classifier.svm.simple_error + 0.02)
        return sum_

    def ada_boost(self):
        _T = len(self.rejecters)
        eps_vec = np.empty(_T)
        alpha_vec = np.empty(_T)
        m = self.num_examples
        _D = np.ones(m) / m
        for t in range(_T):
            example_indexes = []
            for i, d in enumerate(_D):
                if np.random.binomial(1, d) == 1:
                    example_indexes.append(i)
            h = self.rejecters[t]
            t_examples = h.examples
            h.examples = h.examples[example_indexes]
            h.learn()
            h.examples = t_examples
            def bool_to_sign(f): return f * 2 - 1
            classifing_results = bool_to_sign(h.classify_vec(t_examples[:, :-1]))
            eps_vec[t] = (_D * np.abs(classifing_results - t_examples[:, -1])).sum() / 2
            alpha_vec[t] = np.log((1 - eps_vec[t]) / eps_vec[t]) / 2
            _D *= np.exp(-alpha_vec[t]*t_examples[:, -1]*classifing_results)
            _D /= _D.sum()
        return alpha_vec