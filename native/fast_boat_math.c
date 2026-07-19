#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>

static PyObject *plackett_luce(PyObject *self, PyObject *args) {
    PyObject *input;
    if (!PyArg_ParseTuple(args, "O", &input)) {
        return NULL;
    }
    PyObject *sequence = PySequence_Fast(input, "probabilities must be a sequence");
    if (sequence == NULL) {
        return NULL;
    }
    if (PySequence_Fast_GET_SIZE(sequence) != 6) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "six lane probabilities are required");
        return NULL;
    }

    double probabilities[6];
    for (int lane = 0; lane < 6; lane++) {
        probabilities[lane] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(sequence, lane));
        if (PyErr_Occurred()) {
            Py_DECREF(sequence);
            return NULL;
        }
    }
    Py_DECREF(sequence);

    PyObject *result = PyTuple_New(120);
    if (result == NULL) {
        return NULL;
    }
    Py_ssize_t index = 0;
    for (int first = 0; first < 6; first++) {
        const double p_first = probabilities[first];
        const double after_first = fmax(1e-9, 1.0 - p_first);
        for (int second = 0; second < 6; second++) {
            if (second == first) {
                continue;
            }
            const double p_second = probabilities[second];
            const double after_second = fmax(1e-9, 1.0 - p_first - p_second);
            for (int third = 0; third < 6; third++) {
                if (third == first || third == second) {
                    continue;
                }
                double value = p_first * (p_second / after_first)
                    * (probabilities[third] / after_second);
                if (!isfinite(value)) {
                    value = 0.0;
                } else if (value < 0.0) {
                    value = 0.0;
                } else if (value > 1.0) {
                    value = 1.0;
                }
                PyObject *number = PyFloat_FromDouble(value);
                if (number == NULL) {
                    Py_DECREF(result);
                    return NULL;
                }
                PyTuple_SET_ITEM(result, index++, number);
            }
        }
    }
    return result;
}

static PyMethodDef methods[] = {
    {"plackett_luce", plackett_luce, METH_VARARGS,
     "Calculate all 120 ordered trifecta probabilities."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_fast_boat_math",
    "Native numerical kernels for BOAT RACE simulation.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__fast_boat_math(void) {
    return PyModule_Create(&module);
}
