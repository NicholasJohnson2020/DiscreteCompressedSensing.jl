# The implementation is referencing the wikipedia page
# https://en.wikipedia.org/wiki/Matching_pursuit#The_algorithm

const default_max_iter = 200
const default_tolerance = 1e-6


function SparseArrays.sparsevec(d::DefaultDict, m::Int)
    SparseArrays.sparsevec(collect(keys(d)), collect(values(d)), m)
end


function matching_pursuit_(data::AbstractVector, dictionary::AbstractMatrix,
                           max_iter::Int, tolerance::Float64)
    n_atoms = size(dictionary, 2)

    residual = copy(data)

    xdict = DefaultDict{Int, Float64}(0.)
    for i in 1:max_iter
        if norm(residual) < tolerance
            return sparsevec(xdict, n_atoms)
        end

        # find an atom with maximum inner product
        products = dictionary' * residual
        _, maxindex = findmax(abs.(products))
        maxval = products[maxindex]
        atom = dictionary[:, maxindex]

        # c is the length of the projection of data onto atom
        a = maxval / sum(abs2, atom)  # equivalent to maxval / norm(atom)^2
        residual -= atom * a

        xdict[maxindex] += a
    end
    return sparsevec(xdict, n_atoms)
end


"""
    matching_pursuit(data::Vector, dictionary::AbstractMatrix;
                     max_iter::Int = $default_max_iter,
                     tolerance::Float64 = $default_tolerance)
Find ``x`` such that ``Dx = y`` or ``Dx ≈ y`` where y is `data` and D is `dictionary`.
```
# Arguments
* `max_iter`: Hard limit of iterations
* `tolerance`: Exit when the norm of the residual < tolerance
```
"""
function matching_pursuit(data::AbstractVector, dictionary::AbstractMatrix;
                          max_iter::Int = default_max_iter,
                          tolerance = default_tolerance)

    if tolerance <= 0
        throw(ArgumentError("`tolerance` must be > 0"))
    end

    if max_iter <= 0
        throw(ArgumentError("`max_iter` must be > 0"))
    end

    if size(data, 1) != size(dictionary, 1)
        throw(ArgumentError(
            "Dimensions must match: `size(data, 1)` and `size(dictionary, 1)`."
        ))
    end

    matching_pursuit_(data, dictionary, max_iter, tolerance)
end


"""
    matching_pursuit(data::AbstractMatrix, dictionary::AbstractMatrix;
                     max_iter::Int = $default_max_iter,
                     tolerance::Float64 = $default_tolerance)
Find ``X`` such that ``DX = Y`` or ``DX ≈ Y`` where Y is `data` and D is `dictionary`.
```
# Arguments
* `max_iter`: Hard limit of iterations
* `tolerance`: Exit when the norm of the residual < tolerance
```
"""
function matching_pursuit(data::AbstractMatrix, dictionary::AbstractMatrix;
                          max_iter::Int = default_max_iter,
                          tolerance::Float64 = default_tolerance)
    K = size(dictionary, 2)
    N = size(data, 2)

    X = spzeros(K, N)

    for i in 1:N
        X[:, i] = matching_pursuit(
            vec(data[:, i]),
            dictionary,
            max_iter = max_iter,
            tolerance = tolerance
        )
    end
    return X
end;

# This is an implementation of the K-SVD algorithm.
# The original paper:
# K-SVD: An Algorithm for Designing Overcomplete Dictionaries
# for Sparse Representation
# http://www.cs.technion.ac.il/~freddy/papers/120.pdf

# Variable names are based on the original paper.
# If you try to read the code, I recommend you to see Figure 2 first.
#

#include("matching_pursuit.jl")

const default_sparsity_allowance = 0.9
#const default_max_iter = 200
const default_max_iter_mp = 200

Random.seed!(1234)  # for stability of tests


function error_matrix(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, k::Int)
    # indices = [i for i in 1:size(D, 2) if i != k]
    indices = deleteat!(collect(1:size(D, 2)), k)
    return Y - D[:, indices] * X[indices, :]
end


function init_dictionary(n::Int, K::Int)
    # D must be a full-rank matrix
    D = rand(n, K)
    while rank(D) != min(n, K)
        D = rand(n, K)
    end

    @inbounds for k in 1:K
        D[:, k] ./= norm(@view(D[:, k]))
    end
    return D
end


function ksvd(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix)
    N = size(Y, 2)
    for k in 1:size(X, 1)
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        wₖ = findall(!iszero, xₖ)

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        Eₖ = error_matrix(Y, D, X, k)
        Ωₖ = sparse(wₖ, 1:length(wₖ), ones(length(wₖ)), N, length(wₖ))
        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        U, S, V = svd(Eₖ * Ωₖ, full=true)
        D[:, k] = U[:, 1]
        X[k, wₖ] = V[:, 1] * S[1]
    end
    return D, X
end


"""
    ksvd(Y::AbstractMatrix, n_atoms::Int;
         sparsity_allowance::Float64 = $default_sparsity_allowance,
         max_iter::Int = $default_max_iter,
         max_iter_mp::Int = $default_max_iter_mp)
Run K-SVD that designs an efficient dictionary D for sparse representations,
and returns X such that DX = Y or DX ≈ Y.
```
# Arguments
* `sparsity_allowance`: Stop iteration if the number of zeros in X / the number
    of elements in X > sparsity_allowance.
* `max_iter`: Limit of iterations.
* `max_iter_mp`: Limit of iterations in Matching Pursuit that `ksvd` calls at
    every iteration.
```
"""
function ksvd(Y::AbstractMatrix, n_atoms::Int;
              sparsity_allowance = default_sparsity_allowance,
              max_iter::Int = default_max_iter,
              max_iter_mp::Int = default_max_iter_mp)

    K = n_atoms
    n, N = size(Y)

    if !(0 <= sparsity_allowance <= 1)
        throw(ArgumentError("`sparsity_allowance` must be in range [0,1]"))
    end

    X = spzeros(K, N)  # just for making X global in this function
    max_n_zeros = ceil(Int, sparsity_allowance * length(X))

    # D is a dictionary matrix that contains atoms for columns.
    D = init_dictionary(n, K)  # size(D) == (n, K)

    p = Progress(max_iter)

    for i in 1:max_iter
        X_sparse = matching_pursuit(Y, D, max_iter = max_iter_mp)
        D, X = ksvd(Y, D, Matrix(X_sparse))

        # return if the number of zero entries are <= max_n_zeros
        if sum(iszero, X) > max_n_zeros
            return D, X
        end
        next!(p)
    end
    return D, X
end;
