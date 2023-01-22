# pickletools-ext

## Motivation

`pickle` allows arbitrary code execution
([nelhage.com](https://blog.nelhage.com/2011/03/exploiting-pickle/),
[checkoway.net](https://checkoway.net/musings/pickle/),
[PyTorch issue](https://github.com/pytorch/pytorch/issues/31875))

Can we inspect the pickle file before we unpickle it?
Maybe can we even verify that it is safe to unpickle?

## Related

* [`pickletools` from Python standard library](https://docs.python.org/3/library/pickletools.html)
  ([code](https://github.com/python/cpython/blob/main/Lib/pickletools.py))
* [`explain_pickle.py` from the Sage project](https://doc.sagemath.org/html/en/reference/misc/sage/misc/explain_pickle.html)
  ([code](https://github.com/sagemathinc/smc-sage/blob/master/src/sage/misc/explain_pickle.py))
