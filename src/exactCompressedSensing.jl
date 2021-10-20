GUROBI_ENV = Gurobi.Env()

function regobj(X, Y, s, gamma)
    indices = findall(s .> 0.5)
    n = length(Y)
    denom = 2n
    Xs = X[:, indices]
    alpha = Y - Xs * (inv(I / gamma + Xs' * Xs) * (Xs'* Y))
    val = dot(Y, alpha) / denom
    tmp = X' * alpha
    grad_s = -gamma .* tmp .^ 2 ./ denom
  return val, grad_s
end;

function SparseRegression(X, Y, gamma, k; solver_output=1)
    p=size(X)[2]

    #miop = direct_model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
    miop = direct_model(Gurobi.Optimizer(GUROBI_ENV))
    set_optimizer_attribute(miop, "OutputFlag", solver_output)

    # Optimization variables
    @variable(miop, s[1:p], Bin)
    @variable(miop, t >= 0)
    # Objective
    @objective(miop, Min, t)
    # Constraints
    @constraint(miop, sum(s) <= k)
    s0 = zeros(p)
    s0[1:k] .= 1
    p0, grad_s0 = regobj(X, Y, s0, gamma)
    @constraint(miop, t >= p0 + dot(grad_s0, s - s0))
    cb_calls = Cint[]
    function outer_approximation(cb_data, cb_where::Cint)
        push!(cb_calls, cb_where)
        if cb_where != GRB_CB_MIPSOL && cb_where != GRB_CB_MIPNODE
            return
        end
        if cb_where == GRB_CB_MIPNODE
            resultP = Ref{Cint}()
            GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_STATUS, resultP)
            if resultP[] != GRB_OPTIMAL
                return
            end
        end
        Gurobi.load_callback_variable_primal(cb_data, cb_where)
        s_val = callback_value.(cb_data, s)
        t_val = callback_value(cb_data, t)
        obj, grad_s = regobj(X,Y, s_val, gamma)
        offset = sum(grad_s .* s_val)
        if t_val < obj
            con = @build_constraint( t >= obj + sum(grad_s[j] * s[j] for j=1:p) - offset)
            MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
        end
    end
    MOI.set(miop, MOI.RawParameter("LazyConstraints"), 1)
    MOI.set(miop, Gurobi.CallbackFunction(), outer_approximation)

    optimize!(miop)
    s_opt = JuMP.value.(s)
    s_nonzeros = findall(x -> x>0.5, s_opt)
    beta = zeros(p)
    X_s = X[:, s_nonzeros]

    gamma = real(gamma)
    Y = real(Y)
    X_s = real(X_s)

    beta[s_nonzeros] = gamma * X_s' * (Y - X_s * ((I / gamma + X_s' * X_s) \ (X_s'* Y)))
    return s_opt, beta, s_nonzeros
end;

function exactCompressedSensing(A, b, epsilon; gamma_init=1, gamma_max=1e10)

    (m, n) = size(A)
    gamma = gamma_init * n

    x_full = pinv(A'*A)*A'*b
    full_error = norm(A*x_full-b)^2

    if full_error > epsilon
        return false
    end

    best_support = n
    best_beta = x_full
    while gamma <= gamma_max
        if best_support == 1
            break
        end
        current_support = best_support - 1

        _, current_beta, _ = SparseRegression(A, b, gamma, current_support, solver_output=0)
        current_error = norm(A*current_beta-b)^2

        while current_error > epsilon
            gamma = 2 * gamma
            if gamma > gamma_max
                break
            end
            _, current_beta, _ = SparseRegression(A, b, gamma, current_support, solver_output=0)
            current_error = norm(A*current_beta-b)^2
        end
        if current_error < epsilon
            best_support = current_support
            best_beta = current_beta
        end
    end

    return best_support, best_beta
end;

function exactCompressedSensingBinSearch(A, b, epsilon; gamma_init=1,
    gamma_max=1e10)

    (m, n) = size(A)
    gamma = gamma_init * n

    x_full = pinv(A'*A)*A'*b
    full_error = norm(A*x_full-b)^2

    if full_error > epsilon
        return false
    end

    upper_support = n
    upper_beta = x_full
    upper_gamma = gamma

    lower_support = 1
    lower_gamma = gamma

    while upper_support > lower_support + 1

        current_support = lower_support + (upper_support - lower_support) / 2
        current_support = Int64(floor(current_support))
        current_gamma = upper_gamma

        _, current_beta, _ = SparseRegression(A, b, current_gamma,
                                              current_support, solver_output=0)
        current_error = norm(A*current_beta-b)^2

        while current_error > epsilon
            current_gamma = 2 * current_gamma
            if current_gamma > gamma_max
                lower_support = current_support
                lower_gamma = current_gamma
                break
            end
            println("Solving for support $current_support and gamma $current_gamma.")
            _, current_beta, _ = SparseRegression(A, b, current_gamma,
                                                  current_support,
                                                  solver_output=0)
            current_error = norm(A*current_beta-b)^2
        end

        if current_error < epsilon
            upper_support = current_support
            upper_beta = current_beta
        end
    end

    return upper_support, upper_beta, lower_support
end;

function exactCompressedSensingHeuristic(A, b, epsilon)

    (m, n) = size(A)

    x_full = pinv(A'*A)*A'*b
    full_error = norm(A*x_full-b)^2

    if full_error > epsilon
        return false
    end

    if norm(b)^2 < epsilon
        return 0
    end

    first_index = argmax(abs.(b'*A))[2]
    current_support = [first_index]
    current_mat = A[:, first_index]
    current_x = pinv(current_mat'*current_mat)*current_mat'*b
    current_residual = b-current_mat*current_x
    current_error = norm(current_residual)^2

    while current_error > epsilon

        new_index = argmax(abs.(current_residual'A))[2]
        @assert !(new_index in current_support)
        append!(current_support, new_index)
        current_mat = hcat(current_mat, A[:, new_index])
        current_x = pinv(current_mat'*current_mat)*current_mat'*b
        current_residual = b-current_mat*current_x
        current_error = norm(current_residual)^2

    end

    num_support = length(current_support)
    beta = zeros(n)
    for i=1:num_support
        current_index = current_support[i]
        beta[current_index] = current_x[i]
    end

    return num_support, beta

end;
