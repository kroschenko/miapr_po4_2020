import numpy

def get_etalons(variant):

    variant %= 11

    Vector_1 = numpy.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0])
    Vector_2 = numpy.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    Vector_3 = numpy.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    Vector_4 = numpy.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
    Vector_5 = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    Vector_6 = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Vector_7 = numpy.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    Vector_8 = numpy.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1])

    if variant == 1:
        return numpy.array([Vector_1, Vector_6, Vector_8])

    elif variant == 2:
        return numpy.array([Vector_2, Vector_1, Vector_8])

    elif variant == 3:
        return numpy.array([Vector_3, Vector_2, Vector_8])

    elif variant == 4:
        return numpy.array([Vector_4, Vector_3, Vector_8])

    elif variant == 5:
        return numpy.array([Vector_5, Vector_4, Vector_8])

    elif variant == 6:
        return numpy.array([Vector_6, Vector_5, Vector_8])

    elif variant == 7:
        return numpy.array([Vector_7, Vector_6, Vector_8])

    elif variant == 8:
        return numpy.array([Vector_1, Vector_3, Vector_8])

    elif variant == 9:
        return numpy.array([Vector_2, Vector_4, Vector_8])

    elif variant == 10:
        return numpy.array([Vector_3, Vector_5, Vector_8])

    elif variant == 0: #11
        return numpy.array([Vector_4, Vector_6, Vector_8])

    else:
        return numpy.array([[], [], []])
