include("helperLibrary.jl")

function OMP(A, b, epsilon)
    """
    This function executes orthogonal matching pursuit as described in Section
    2.3 of the accompanying paper.

    :param A: A m-by-n design matrix.
    :param b: An m dimensional vector of observations.
    :param epsilon: A numerical threshold parameter (Float64).

    :return: This function returns two values:
             1) The x vector output of orthogonal matching pursuit.
             2) The cardinality of the output vector (Int64).
    """
    (m, n) = size(A)

    x_full = pinv(A'*A)*A'*b
    full_error = norm(A*x_full-b)^2

    if full_error > epsilon
        return false, nothing
    end

    if norm(b)^2 < epsilon
        return zeros(n), 0
    end

    first_index = argmax(abs.(b'*A))[2]
    current_support = [first_index]
    num_support = 1
    current_mat = A[:, first_index]
    stored_inv = 1 / norm(current_mat)^2
    current_x = stored_inv * current_mat' * b
    current_residual = b-current_mat*current_x
    current_error = norm(current_residual)^2

    # Greedily expand the support until the norm of the residual is less than
    # epsilon
    while current_error > epsilon

        if num_support == n
            break
        end
        scores = abs.(current_residual'A)
        for index in current_support
            scores[index] = -1
        end
        new_index = argmax(scores)[2]
        append!(current_support, new_index)
        num_support += 1
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
