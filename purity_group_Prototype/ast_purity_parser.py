import typing

from astroid import parse
from astroid import nodes as Astroid
import json

sklearn = parse('''
class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init=10,
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="auto",
    ):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm

    def _check_params(self, X):
        # n_init
        if self.n_init <= 0:
            raise ValueError(f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )

        # tol
        self._tol = _tolerance(X, self.tol)

        # algorithm
        if self.algorithm not in ("auto", "full", "elkan"):
            raise ValueError(
                "Algorithm must be 'auto', 'full' or 'elkan', "
                f"got {self.algorithm} instead."
            )

        self._algorithm = self.algorithm
        if self._algorithm == "auto":
            self._algorithm = "full" if self.n_clusters == 1 else "elkan"
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn(
                "algorithm='elkan' doesn't make sense for a single "
                "cluster. Using 'full' instead.",
                RuntimeWarning,
            )
            self._algorithm = "full"

        # init
        if not (
            hasattr(self.init, "__array__")
            or callable(self.init)
            or (isinstance(self.init, str) and self.init in ["k-means++", "random"])
        ):
            raise ValueError(
                "init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{self.init}' instead."
            )

        if hasattr(self.init, "__array__") and self._n_init != 1:
            warnings.warn(
                "Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self._n_init}.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._n_init = 1

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters."""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}."
            )
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}."
            )

    def _check_test_data(self, X):
        X = self._validate_data(
            X,
            accept_sparse="csr",
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return X

    def _check_mkl_vcomp(self, X, n_samples):
        """Warns when vcomp and mkl are both present"""
        # The BLAS call inside a prange in lloyd_iter_chunked_dense is known to
        # cause a small memory leak when there are less chunks than the number
        # of available threads. It only happens when the OpenMP library is
        # vcomp (microsoft OpenMP) and the BLAS library is MKL. see #18653
        if sp.issparse(X):
            return

        active_threads = int(np.ceil(n_samples / CHUNK_SIZE))
        if active_threads < self._n_threads:
            modules = threadpool_info()
            has_vcomp = "vcomp" in [module["prefix"] for module in modules]
            has_mkl = ("mkl", "intel") in [
                (module["internal_api"], module.get("threading_layer", None))
                for module in modules
            ]
            if has_vcomp and has_mkl:
                if not hasattr(self, "batch_size"):  # KMeans
                    warnings.warn(
                        "KMeans is known to have a memory leak on Windows "
                        "with MKL, when there are less chunks than available "
                        "threads. You can avoid it by setting the environment"
                        f" variable OMP_NUM_THREADS={active_threads}."
                    )
                else:  # MiniBatchKMeans
                    warnings.warn(
                        "MiniBatchKMeans is known to have a memory leak on "
                        "Windows with MKL, when there are less chunks than "
                        "available threads. You can prevent it by setting "
                        f"batch_size >= {self._n_threads * CHUNK_SIZE} or by "
                        "setting the environment variable "
                        f"OMP_NUM_THREADS={active_threads}"
                    )

    def _init_centroids(self, X, x_squared_norms, init, random_state, init_size=None):
        """Compute the initial centroids.
        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.
        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.
        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.
        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).
        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]

        if isinstance(init, str) and init == "k-means++":
            centers, _ = _kmeans_plusplus(
                X,
                n_clusters,
                random_state=random_state,
                x_squared_norms=x_squared_norms,
            )
        elif isinstance(init, str) and init == "random":
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = X[seeds]
        elif hasattr(init, "__array__"):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(centers, dtype=X.dtype, copy=False, order="C")
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
            .. versionadded:: 0.20
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(
            X,
            accept_sparse="csr",
            dtype=[np.float64, np.float32],
            order="C",
            copy=self.copy_x,
            accept_large_sparse=False,
        )

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        self._n_threads = _openmp_effective_n_threads()

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype, copy=True, order="C")
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, "__array__"):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "full":
            kmeans_single = _kmeans_single_lloyd
            self._check_mkl_vcomp(X, X.shape[0])
        else:
            kmeans_single = _kmeans_single_elkan

        best_inertia, best_labels = None, None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init, random_state=random_state
            )
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=self.verbose,
                tol=self._tol,
                x_squared_norms=x_squared_norms,
                n_threads=self._n_threads,
            )

            # determine if these results are the best so far
            # we chose a new run if it has a better inertia and the clustering is
            # different from the best so far (it's possible that the inertia is
            # slightly better even if the clustering is the same with potentially
            # permuted labels, due to rounding errors)
            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster
        centers. Note that even if X is sparse, the array returned by
        `transform` will typically be dense.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.
        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """Guts of transform method; no input validation."""
        return euclidean_distances(X, self.cluster_centers_)

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return _labels_inertia_threadpool_limit(
            X, sample_weight, x_squared_norms, self.cluster_centers_, self._n_threads
        )[0]

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.
        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return -_labels_inertia_threadpool_limit(
            X, sample_weight, x_squared_norms, self.cluster_centers_, self._n_threads
        )[1]

    def _more_tags(self):
        return {
            "_xfail_checks": {
                "check_sample_weights_invariance": (
                    "zero sample_weight is not equivalent to removing samples"
                ),
            },
        }
''')

module = parse('''
class A:
    def __init__(self):
        self.a = 1
        input("Schreib was :)")
        print('alles erfolgreich initialisiert')

    def func(self, arg):
        self.a = 5
        if arg < random_state:
            x = self.a
            raise ValueError("hello there")
        elif arg > 10 and arg < 20:
            x = self.a + 7
        else:
            x = self.a * 6
        x = random_state
        f(self)
        return self.a

aObj = A()

def f(pA):
    file = open("myfile.txt", "a")
    a.a = 7

f(aObj)
''')

''' Lists to store function properties while traversing the ast '''
call_prop_list: typing.List[typing.Dict[str, str]] = []
state_read_prop_list: typing.List[typing.Dict[str, str]] = []
state_write_prop_list: typing.List[typing.Dict[str, str]] = []
input_read_prop_list: typing.List[typing.Dict[str, str]] = []
output_write_prop_list: typing.List[typing.Dict[str, str]] = []
error_prop_list: typing.List[typing.Dict[str, str]] = []
random_prop_list: typing.List[typing.Dict[str, str]] = []

''' Save entry in list for the case where a function calls another function '''
def create_call_prop(fnc: str, callee: str):
    new_prop = {"function": fnc, "callee": callee}
    call_prop_list.append(new_prop)

''' Save entry in list for the case where a function reads from a state '''
def create_state_read_prop(fnc: str, attr: str, obj: str):
    new_prop = {"function": fnc, "attribute": attr, "object": obj}
    state_read_prop_list.append(new_prop)

''' Save entry in list for the case where a function write to the state of an object '''
def create_state_write_prop(fnc: str, attr: str, obj: str, value: str):
    new_prop = {"function": fnc, "attribute": attr, "object": obj, "value": value}
    state_write_prop_list.append(new_prop)

''' Save entry in list for the case where a function reads input from file or console '''
def create_input_read_prop(fnc: str, source: str):
    new_prop = {"function": fnc, "source": source}
    input_read_prop_list.append(new_prop)

''' Save entry in list for the case where a function writes to the file or console '''
def create_output_write_prop(fnc: str, target: str, text: str):
    new_prop = {"function": fnc, "target": target, "text": text}
    output_write_prop_list.append(new_prop)

''' Save entry in list for the case where a function raises an exception '''
def create_error_prop(fnc: str, cond: str, error: str, msg: str):
    new_prop = {"function": fnc, "condition": cond, "error": error, "message": msg}
    error_prop_list.append(new_prop)

''' Save entry in list for the case where a function uses randomness '''
def create_random_prop(fnc: str, source: str):
    new_prop = {"function": fnc, "source": source}
    random_prop_list.append(new_prop)

''' Serialize all lists '''
def serialize_lists():
    file = open("parsed_data.json", "w")
    file.write(json.dumps(call_prop_list))
    file.write(json.dumps(state_read_prop_list))
    file.write(json.dumps(state_write_prop_list))
    file.write(json.dumps(input_read_prop_list))
    file.write(json.dumps(output_write_prop_list))
    file.write(json.dumps(error_prop_list))
    file.write(json.dumps(random_prop_list))
    file.close()

''' Print all lists '''
def print_lists():
    for e in call_prop_list:
        print(f'\033[93mFunction "{e.get("function")}" references other function "{e.get("callee")}", inheriting all its properties\033[0m')

    for e in state_read_prop_list:
        print(f'\033[92mUnpure function "{e.get("function")}" due to state read: uses "{e.get("attribute")}" of "{e.get("object")}"\033[0m')

    for e in state_write_prop_list:
        print(f'\033[92mFunction "{e.get("function")}" has state side effect: writes to "{e.get("attribute")}" of "{e.get("object")}"\033[0m')

    for e in input_read_prop_list:
        print(f'\033[96mUnpure function "{e.get("function")}" due to read from "{e.get("target")}"\033[0m')

    for e in output_write_prop_list:
        print(f'\033[96mFunction "{e.get("function")}" has output side effect: prints "{e.get("text")}" to {e.get("target")}\033[0m')

    for e in error_prop_list:
        print(f'\033[95mFunction "{e.get("function")}" may terminate abnormally: raises "{e.get("error")}" under condition "{e.get("condition")}" and prints "{e.get("message")}" to console\033[0m')

    for e in random_prop_list:
        print(f'\033[94mUnpure function "{e.get("function")}" due to randomness read: uses "{e.get("source")}"\033[0m')

def visitAst(ast):
    if isinstance(ast, Astroid.Module):
        # handle module
        for i in range(len(ast.body)):
            visitAst(ast.body[i])

    if isinstance(ast, Astroid.FunctionDef) or isinstance(ast, Astroid.AsyncFunctionDef):
        # handle function
        for i in range(len(ast.body)):
            visitAst(ast.body[i])
    if isinstance(ast, Astroid.ClassDef):
        # handle class
        for i in range(len(ast.instance_attrs)):
            visitAst(ast.instance_attrs.get(i))
        for i in range(len(ast.keywords)):
            visitAst(ast.keywords[i])
        for i in range(len(ast.locals)):
            visitAst(ast.locals.get(i))
        for i in range(len(ast.bases)):
            visitAst(ast.bases[i])
        for i in range(len(ast.body)):
            visitAst(ast.body[i])

    if isinstance(ast, Astroid.Expr):
        # handle expression
        visitAst(ast.value)
    if isinstance(ast, Astroid.If):
        # handle condition
        visitAst(ast.test)
        # handle body
        for i in range(len(ast.body)):
            if ast.body[i] is not None:
                visitAst(ast.body[i])
        # handle else, which might be an if again
        for i in range(len(ast.orelse)):
            if ast.orelse[i] is not None:
                visitAst(ast.orelse[i])

    if isinstance(ast, Astroid.Compare):
        # handle operand
        visitAst(ast.left)
        # handle ops
        for i in range(len(ast.ops)):
            visitAst(ast.ops[i])

    if isinstance(ast, tuple) and isinstance(ast[0], str) and isinstance(ast[1], Astroid.NodeNG):
        # handle operand
        visitAst(ast[1])

    if isinstance(ast, Astroid.BoolOp):
        # handle operands
        for i in range(len(ast.values)):
            if ast.values[i] is not None:
                visitAst(ast.values[i])

    if isinstance(ast, Astroid.BinOp):
        # handle operands
        visitAst(ast.left)
        visitAst(ast.right)
    if isinstance(ast, Astroid.UnaryOp):
        # handle operand
        visitAst(ast.operand)

    if isinstance(ast, Astroid.Raise):
        # handle raise
        # if isinstance(ast.exc.args[0], Astroid.JoinedStr):
        #     string = ""
        #     for i in range(len(ast.exc.args[0].values)):
        #         if isinstance(ast.exc.args[0].values[i], Astroid.Const):
        #             string += " " + ast.exc.args[0].values[i].value
        #         elif isinstance(ast.exc.args[0].values[i], str):
        #             string += " " + ast.exc.args[0].values[i]
        # else:
        #     string = ast.exc.args[0].value
        create_error_prop(inferFunction(ast), "", ast.exc.func.name, concatArgs(ast.exc.args)) #TODO infer condition under which exception is raised

    if isinstance(ast, Astroid.Const):
        # handle const
        visitAst(ast.value)

    if isinstance(ast, Astroid.Call):
        # handle expression
        enclosing = inferFunction(ast)

        if isinstance(ast.func, Astroid.FunctionDef) or isinstance(ast.func, Astroid.AsyncFunctionDef):
            if ast.func.name == "print":
                create_output_write_prop(enclosing, "console", concatArgs(ast.args))
            elif ast.func.name == "input":
                create_input_read_prop(enclosing, "console")
                if len(ast.args) > 0:
                    argsJoined = ""
                    for i in range(len(ast.args)):
                        argsJoined += " " + ast.args[i].value
                    argsJoined = argsJoined.strip()
                    create_output_write_prop(enclosing, "console", argsJoined)
            elif ast.func.name == "open":
                if ast.args[1].value == "r":
                    create_input_read_prop(enclosing, str(ast.args[0].value))
                if ast.args[1].value in "axw":
                    argsJoined = ""
                    for i in range(1, len(ast.args)):
                        argsJoined += " " + ast.args[i].value
                    argsJoined = argsJoined.strip()
                    create_output_write_prop(enclosing, str(ast.args[0].value), argsJoined)
            elif enclosing is not None:
                create_call_prop(enclosing, ast.func.name)
            else:
                print(f'\033[93mUnknown!!\033[0m')
        elif isinstance(ast.func, Astroid.Attribute) and ast.func.attrname == 'warn':
            string = ast.args[0] if len(ast.args) > 0 else ""
            create_output_write_prop(enclosing, "console", string) #TODO: is warn really an output or are they just collected??
        elif isinstance(ast.func, Astroid.Name):
            if ast.func.name == 'print':
                create_output_write_prop(enclosing, "console", concatArgs(ast.args))
            else:
                if ast.func != enclosing: #TODO: Check for mutual recursion in set
                    create_call_prop(enclosing, ast.func.name)
                    #TODO: also check if function in parameter place or inner function can be a problem (they are not necessary executed)!!

        for i in range(len(ast.args)):
            visitAst(ast.args[i])
        for i in range(len(ast.keywords)):
            visitAst(ast.keywords[i])

    if isinstance(ast, Astroid.Assign):
        # handle expression
        visitAst(ast.value)

        for i in range(len(ast.targets)):
            visitAst(ast.targets[i])

    if isinstance(ast, Astroid.AugAssign):
        # handle expression
        visitAst(ast.target)
        visitAst(ast.value)

    if isinstance(ast, Astroid.AssignAttr):
        # handle attribute assignment
        visitAst(ast.expr)
        create_state_write_prop(inferFunction(ast), ast.attrname, ast.expr.name, ast.expr)

    if isinstance(ast, Astroid.Attribute):
        # handle attribute read
        if isinstance(ast.expr, Astroid.Name):
            name = ast.expr.name
        elif isinstance(ast.expr, Astroid.Call):
            name = ast.expr.func
        else:
            name = "NONAME"
        create_state_read_prop(inferFunction(ast), ast.attrname, ast.expr.name if hasattr(ast.expr, "name") else "")

    if isinstance(ast, Astroid.Return):
        # handle return
        visitAst(ast.value)

    if isinstance(ast, Astroid.Keyword):
        # handle keyword
        # visitAst(ast.arg) TODO is this necessary? since we can only set the parameters...?
        visitAst(ast.value)

    if isinstance(ast, Astroid.For) or isinstance(ast, Astroid.AsyncFor):
        # handle for
        visitAst(ast.iter)
        visitAst(ast.target)
        visitAst(ast.type_annotation)
        for i in range(len(ast.body)):
            visitAst(ast.body[i])
        for i in range(len(ast.orelse)):
            visitAst(ast.orelse[i])

    if isinstance(ast, tuple) and isinstance(ast[0], str) and isinstance(ast[1], list):
        # handle list
        visitAst(ast[1])

    if isinstance(ast, list):
        # handle module
        for i in range(len(ast)):
            visitAst(ast[i])

    #Here begins Bitas part

    if isinstance(ast, Astroid.Compare):
        # handle compare
        visitAst(ast.left)
        for i in range(len(ast.ops)):
            visitAst(ast.ops[i])

    if isinstance(ast, Astroid.BaseContainer):
        # handle module

        for i in range(len(ast.elts)):
            visitAst(ast.elts[i])

    if isinstance(ast, Astroid.Arguments):
        # handle Arguments
        for i in range(len(ast.args)):
            visitAst(ast.args[i])
        for i in range(len(ast.defaults)):
            visitAst(ast.defaults[i])
        for i in range(len(ast.kwonlyargs)):
            visitAst(ast.kwonlyargs[i])
        for i in range(len(ast.posonlyargs)):
            visitAst(ast.posonlyargs[i])
        for i in range(len(ast.kw_defaults)):
            visitAst(ast.kw_defaults[i])
        for i in range(len(ast.annotations)):
            visitAst(ast.annotations[i])
        for i in range(len(ast.posonlyargs_annotations)):
            visitAst(ast.posonlyargs_annotations[i])
        for i in range(len(ast.kwonlyargs_annotations)):
            visitAst(ast.kwonlyargs_annotations[i])
        for i in range(len(ast.type_comment_args)):
            visitAst(ast.type_comment_args[i])

    if isinstance(ast, Astroid.AnnAssign):
        # handle module
        visitAst(ast.target)
        visitAst(ast.annotation)
        visitAst(ast.value)

    if isinstance(ast, Astroid.Assert):
        # handle module
        visitAst(ast.test)
        visitAst(ast.fail)

    if isinstance(ast, Astroid.AssignName):
        # handle module
        visitAst(ast.name)

    if isinstance(ast, Astroid.Await):
        # handle module
        visitAst(ast.value)

    if isinstance(ast, Astroid.Comprehension):
        # handle module
        visitAst(ast.target)
        visitAst(ast.iter)
        for i in range(len(ast.ifs)):
            visitAst(ast.ifs[i])

    if isinstance(ast, Astroid.Const):
        # handle module
        visitAst(ast.value)

    if isinstance(ast, Astroid.Decorators):
        # handle module
        for i in range(len(ast.nodes)):
            visitAst(ast.nodes[i])

    if isinstance(ast, Astroid.DelAttr):
        # handle module
        visitAst(ast.expr)

    if isinstance(ast, Astroid.Delete):   #<<<<------- state side effect?
        # handle module
        for i in range(len(ast.targets)):
            visitAst(ast.targets[i])

    if isinstance(ast, Astroid.Dict):
        # handle module
        for i in range(len(ast.items)):
            visitAst(ast.items[i])

    if isinstance(ast, Astroid.DictComp):
        # handle module
        for i in range(len(ast.locals)):
            visitAst(ast.locals[i])

    if isinstance(ast, Astroid.EvaluatedObject):
        # handle module
        visitAst(ast.original)
        visitAst(ast.value)

    if isinstance(ast, Astroid.ExceptHandler):
        # handle module
        visitAst(ast.type)
        visitAst(ast.name)
        for i in range(len(ast.body)):
            visitAst(ast.body[i])

    if isinstance(ast, Astroid.FormattedValue):
        # handle module
        visitAst(ast.format_spec)
        visitAst(ast.value)

    if isinstance(ast, Astroid.GeneratorExp):
        # handle module
        for i in range(len(ast.locals)):
            visitAst(ast.locals[i])

    if isinstance(ast, Astroid.IfExp):
        # handle module
        visitAst(ast.test)
        visitAst(ast.body)
        visitAst(ast.orelse)

    if isinstance(ast, Astroid.JoinedStr):
        # handle module
        for i in range(len(ast.values)):
            visitAst(ast.values[i])

    if isinstance(ast, Astroid.Lambda):
        # handle module
        visitAst(ast.args)

        for i in range(len(ast.locals)):
            visitAst(ast.locals.get(i))
        for i in range(len(ast.body)):
            visitAst(ast.body[i])

    if isinstance(ast, Astroid.ListComp):
        # handle module
        for i in range(len(ast.locals)):
            visitAst(ast.locals.get(i))

    if isinstance(ast, Astroid.LocalsDictNodeNG):
        # handle module
        for i in range(len(ast.locals)):
            visitAst(ast.locals.get(i))

    if isinstance(ast, Astroid.Match):
        # handle module
        visitAst(ast.subject)

        for i in range(len(ast.cases)):
            visitAst(ast.cases[i])

    if isinstance(ast, Astroid.MatchAs):
        # handle module
        visitAst(ast.pattern)
        visitAst(ast.name)

    if isinstance(ast, Astroid.MatchCase):
        # handle module
        visitAst(ast.pattern)
        visitAst(ast.guard)

        for i in range(len(ast.body)):
            visitAst(ast.body[i])

    if isinstance(ast, Astroid.MatchClass):
        # handle module
        visitAst(ast.cls)
        for i in range(len(ast.kwd_patterns)):
            visitAst(ast.kwd_patterns[i])
        for i in range(len(ast.patterns)):
            visitAst(ast.patterns[i])

    if isinstance(ast, Astroid.MatchMapping):
        # handle module
        visitAst(ast.rest)
        for i in range(len(ast.keys)):
            visitAst(ast.keys[i])
        for i in range(len(ast.patterns)):
            visitAst(ast.patterns[i])

    if isinstance(ast, Astroid.MatchOr):
        # handle module
        for i in range(len(ast.patterns)):
            visitAst(ast.patterns[i])

    if isinstance(ast, Astroid.MatchSequence):
        # handle module
        for i in range(len(ast.patterns)):
            visitAst(ast.patterns[i])

    if isinstance(ast, Astroid.MatchStar):
        # handle module
        visitAst(ast.name)

    if isinstance(ast, Astroid.MatchValue):
        # handle module
        visitAst(ast.name)

    if isinstance(ast, Astroid.NamedExpr):
        # handle module
        visitAst(ast.target)
        visitAst(ast.value)

    if isinstance(ast, Astroid.Set):
        # handle module
        for i in range(len(ast.elts)):
            visitAst(ast.elts[i])

    if isinstance(ast, Astroid.SetComp):
        # handle module
        for i in range(len(ast.locals)):
            visitAst(ast.locals[i])

    if isinstance(ast, Astroid.Slice):
        # handle module
        visitAst(ast.lower)
        visitAst(ast.upper)
        visitAst(ast.step)

    if isinstance(ast, Astroid.Starred):
        # handle module
        visitAst(ast.value)
        visitAst(ast.ctx)

    if isinstance(ast, Astroid.Subscript):
        # handle module
        visitAst(ast.value)
        visitAst(ast.slice)
        visitAst(ast.ctx)

    if isinstance(ast, Astroid.TryExcept):
        # handle module
        for i in range(len(ast.body)):
            visitAst(ast.body[i])
        for i in range(len(ast.orelse)):
            visitAst(ast.orelse[i])
        for i in range(len(ast.handlers)):
            visitAst(ast.handlers[i])

    if isinstance(ast, Astroid.TryFinally):
        # handle module
        for i in range(len(ast.body)):
            visitAst(ast.body[i])
        for i in range(len(ast.finalbody)):
            visitAst(ast.finalbody[i])

    if isinstance(ast, Astroid.While):
        # handle module
        visitAst(ast.test)

        for i in range(len(ast.body)):
            visitAst(ast.body[i])
        for i in range(len(ast.orelse)):
            visitAst(ast.orelse[i])

    if isinstance(ast, Astroid.With) or isinstance(ast, Astroid.AsyncWith):
        # handle module
        visitAst(ast.type_annotation)
        for i in range(len(ast.items)):
            visitAst(ast.items[i])
        for i in range(len(ast.body)):
            visitAst(ast.body[i])

    if isinstance(ast, Astroid.Yield):
        # handle module
        visitAst(ast.value)

    #######################
    ##### Randomness ######
    #######################

    if isinstance(ast, Astroid.Name) and "random" in str(ast.name):
        # handle attribute read
        create_random_prop(inferFunction(ast), ast.name)


def inferFunction(ast):
    if ast is None:
        return None
    if isinstance(ast, Astroid.FunctionDef):
        return ast.name
    else:
        return inferFunction(ast.parent)


def inferClass(ast):
    if ast is None:
        return None
    if isinstance(ast, Astroid.ClassDef):
        return ast.name
    else:
        return inferClass(ast.parent)


def concatJoinedStr(strings):
    if strings is Astroid.JoinedStr:
        strings = strings.values
        string = ""
        for i in range(len(strings)):
            string += strings[i]
        return string.strip()
    elif strings is Astroid.Const:
        return strings.value


def _concatJoinedString(strings):
    string = ""
    for i in range(len(strings.values)):
        string += strings.values[i].as_string()
    return string.strip()


def concatArgs(strings):
    string = ""
    for i in range(len(strings)):
        # if isinstance(strings[i], Astroid.JoinedStr):
        #     string += " " + _concatJoinedString(strings[i])
        # elif isinstance(strings[i], Astroid.Const):
        #     string += " " + strings[i].value

        string += " " + strings[i].as_string()
    return string.strip()


if __name__ == '__main__':
    visitAst(sklearn)
    print_lists()