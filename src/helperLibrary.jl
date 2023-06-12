function update_inverse(A, inv_ATA, a_i)
    """
    This function computes the pseudo inverse of the matrix
    [A'A A'a_i; a_i'A a_i'a_i] by performing a rank one update on the matrix
    inv_ATA, which is the pseudo inverse of A'A.

    :param A: A m-by-t matrix.
    :param inv_ATA: A t-by-t matrix that is pseudo inverse of A'A.
    :param a_i: A m-dimensional vector.

    :return: A (t+1)-by(t+1) matrix.
    """
    comp = a_i'*a_i - a_i'*A*inv_ATA*A'*a_i
    if typeof(inv_ATA) == Float64
        n = 1
    else
        n = size(inv_ATA)[1]
    end
    new_inv = zeros(n+1, n+1)
    new_inv[(n+1), (n+1)] = 1/comp
    if n == 1
        temp = (inv_ATA*A'*a_i)[1]
        new_inv[1, 1] = inv_ATA + temp^2 / comp
        new_inv[1, 2] = -temp / comp
        new_inv[2, 1] = -temp' / comp
    else
        temp = inv_ATA*A'*a_i
        new_inv[1:n, 1:n] = inv_ATA + (temp*temp') / comp
        new_inv[1:n, n+1] = -temp / comp
        new_inv[n+1, 1:n] = -temp' / comp
    end

    return new_inv
end;


function roundSolution(x, A, b, epsilon; lower_bound=1)
    """
    This function performs greedy rounding on the input vector x according to
    Algorithm 1 in Section 5.1.2 of the accompanying paper.

    :param x: An n dimensional vector.
    :param A: A m-by-n design matrix.
    :param b: An m dimensional vector of observations.
    :param epsilon: A numerical threshold parameter (Float64).
    :param lower_bound: A lower bound on the optimial compressed sensing
                        objective value (Float64).

    :return: This function returns two values. The first is an n dimensional
             vector that corresponds to the rounded version of the vector x. The
             second is the cardinality of the rounded vector.
    """
    (m, n) = size(A)
    ordering = sortperm(abs.(x), rev=true)

    @assert lower_bound >= 0
    num_support = Int(ceil(lower_bound))
    current_support = [ordering[i] for i in 1:num_support]
    current_mat = zeros(m, num_support)
    for i in 1:num_support
        current_mat[:, i] = A[:, current_support[i]]
    end
    if lower_bound >= 2
        stored_inv = pinv(current_mat'*current_mat)
    else
        stored_inv = 1 / norm(current_mat)^2
    end
    current_x = stored_inv * current_mat' * b
    current_residual = b-current_mat*current_x
    current_error = norm(current_residual)^2

    # Greedily expand the support until the norm of the residual is less than
    # epsilon
    while current_error > epsilon

        if num_support == n
            break
        end
        num_support += 1
        new_index = ordering[num_support]
        append!(current_support, new_index)
        stored_inv = update_inverse(current_mat, stored_inv, A[:, new_index])
        current_mat = hcat(current_mat, A[:, new_index])
        current_x = stored_inv*current_mat'*b
        current_residual = b-current_mat*current_x
        current_error = norm(current_residual)^2

    end

    beta = zeros(n)
    for i=1:num_support
        current_index = current_support[i]
        beta[current_index] = current_x[i]
    end

    return beta, num_support

end;
