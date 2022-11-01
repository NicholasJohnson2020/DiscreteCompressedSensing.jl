function solveSubproblemPrimalL2(A, b, epsilon, gamma;
    solver_output=0, round_solution=true, zero_indices=[], one_indices=[])

    (m, n) = size(A)

    unique_indices = unique([zero_indices; one_indices])
    num_zeros = size(zero_indices)[1]
    num_ones = size(one_indices)[1]
    free_indices = collect(1:n)[(!in).(collect(1:n), Ref(unique_indices))]

    @assert num_ones + num_zeros <= n
    @assert size(unique_indices)[1] == num_ones + num_zeros

    model = Model(with_optimizer(Gurobi.Optimizer))
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, x[i=1:n])
    @variable(model, abs_x[i in free_indices]>=0)
    @variable(model, residual[i=1:m])
    @variable(model, t>=0)

    @constraint(model, [i in free_indices], abs_x[i] >= x[i])
    @constraint(model, [i in free_indices], -abs_x[i] <= x[i])
    @constraint(model, residual .== A * x .- b)

    @constraint(model, [i in zero_indices], x[i] == 0)

    @constraint(model, sum(residual[i]^2 for i=1:m) <= epsilon)
    @constraint(model, sum(x[i]^2 for i in one_indices) <= t)

    @objective(model, Min,
               2 / sqrt(gamma) * sum(abs_x[i] for i in free_indices) + t / gamma)

    optimize!(model)

    opt_x = value.(x)
    opt_z = zeros(n)
    for i in one_indices
        opt_z[i] = 1
    end
    for i in free_indices
        opt_z[i] = abs(opt_x[i]) / sqrt(gamma)
    end

    if round_solution
        rounded_x, num_support = roundSolution(opt_x, A, b, epsilon,
                                               norm_function="L2")

        return rounded_x, num_support + norm(rounded_x)^2 / gamma,
               opt_x, opt_z, objective_value(model) + size(one_indices)[1]
    else
        return opt_x, opt_z, objective_value(model) + size(one_indices)[1]
    end

end;

function solveSubproblemDualL2(A, b, epsilon, gamma;
    solver_output=0, zero_indices=[], one_indices=[])

    (m, n) = size(A)
    all_indices = collect(1:n)
    free_indices = all_indices[(!in).(all_indices, Ref(zero_indices))]
    free_indices = free_indices[(!in).(free_indices, Ref(one_indices))]

    model = Model(with_optimizer(Gurobi.Optimizer))
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, nu[j=1:m])
    @variable(model, t)
    @variable(model, aux[i=1:n])

    @constraint(model, [t; nu] in SecondOrderCone())
    @constraint(model, [i=1:n], aux[i] == nu'*A[:, i])

    @constraint(model, upper[i in free_indices], aux[i] <= 2/sqrt(gamma))
    @constraint(model, lower[i in free_indices], aux[i] >= -2/sqrt(gamma))

    @objective(model, Max, b'*nu - sqrt(epsilon) * t - gamma / 4 * sum(aux[i]^2 for i in one_indices))

    optimize!(model)

    opt_val = objective_value(model) + size(one_indices)[1]

    return opt_val, value.(nu)

end;

function solveSubproblemPrimalL1(A, b, epsilon, gamma;
    solver_output=0, round_solution=true, zero_indices=[], one_indices=[])

    (m, n) = size(A)

    unique_indices = unique([zero_indices; one_indices])
    num_zeros = size(zero_indices)[1]
    num_ones = size(one_indices)[1]
    free_indices = collect(1:n)[(!in).(collect(1:n), Ref(unique_indices))]

    @assert num_ones + num_zeros <= n
    @assert size(unique_indices)[1] == num_ones + num_zeros

    model = Model(with_optimizer(Gurobi.Optimizer))
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, x[i=1:n])
    @variable(model, abs_x[i in free_indices]>=0)
    @variable(model, residual[i=1:m])
    @variable(model, t>=0)

    @constraint(model, [i in free_indices], abs_x[i] >= x[i])
    @constraint(model, [i in free_indices], -abs_x[i] <= x[i])
    @constraint(model, residual .>= A * x .- b)
    @constraint(model, residual .>= - A * x .+ b)

    @constraint(model, [i in zero_indices], x[i] == 0)

    @constraint(model, sum(residual[i] for i=1:m) <= epsilon)
    @constraint(model, sum(x[i]^2 for i in one_indices) <= t)

    @objective(model, Min,
               2 / sqrt(gamma) * sum(abs_x[i] for i in free_indices) + t / gamma)

    optimize!(model)

    opt_x = value.(x)
    opt_z = zeros(n)
    for i in one_indices
        opt_z[i] = 1
    end
    for i in free_indices
        opt_z[i] = abs(opt_x[i]) / sqrt(gamma)
    end

    if round_solution
        rounded_x, num_support = roundSolution(opt_x, A, b, epsilon,
                                               norm_function="L1")

        return rounded_x, num_support + norm(rounded_x)^2 / gamma,
               opt_x, opt_z, objective_value(model) + size(one_indices)[1]
    else
        return opt_x, opt_z, objective_value(model) + size(one_indices)[1]
    end

end;

function solveSubproblemDualL1(A, b, epsilon, gamma;
    solver_output=0, zero_indices=[], one_indices=[])

    (m, n) = size(A)
    all_indices = collect(1:n)
    free_indices = all_indices[(!in).(all_indices, Ref(zero_indices))]
    free_indices = free_indices[(!in).(free_indices, Ref(one_indices))]

    model = Model(with_optimizer(Gurobi.Optimizer))
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, nu[j=1:m])
    @variable(model, t)
    @variable(model, aux[i=1:n])

    @constraint(model, [i=1:n], aux[i] == nu'*A[:, i])

    @constraint(model, upper[i in free_indices], aux[i] <= 2/sqrt(gamma))
    @constraint(model, lower[i in free_indices], aux[i] >= -2/sqrt(gamma))
    @constraint(model, [j=1:m], t >= nu[j])
    @constraint(model, [j=1:m], t >= -nu[j])

    @objective(model, Max, -b'*nu - epsilon * t - gamma / 4 * sum(aux[i]^2 for i in one_indices))

    optimize!(model)

    opt_val = objective_value(model) + size(one_indices)[1]
    upper_vals = dual.(upper)
    lower_vals = dual.(lower)

    return opt_val, value.(nu), upper_vals, lower_vals

end;

function solveSubproblemL1OSQP(A, b, epsilon, gamma;
    solver_output=0, round_solution=true, zero_indices=[], one_indices=[])

    (m, n) = size(A)

    unique_indices = unique([zero_indices; one_indices])
    num_zeros = size(zero_indices)[1]
    num_ones = size(one_indices)[1]
    free_indices = collect(1:n)[(!in).(collect(1:n), Ref(unique_indices))]

    @assert num_ones + num_zeros <= n
    @assert size(unique_indices)[1] == num_ones + num_zeros

    P = zeros(2 * n + m, 2 * n + m)
    for index in one_indices
        P[index, index] = 2 / gamma
    end
    P = sparse(P)
    q = zeros(2 * n + m)
    for index in free_indices
        q[index + n] = 2 / sqrt(gamma)
    end
    A_mat = zeros(2 * m + 3  * n + 1, 2 * n + m)
    A_mat[1:m, 1:n] = A
    A_mat[1:m, (2 * n + 1):(2 * n + m)] = -1 * Matrix(I, m, m)
    A_mat[(m + 1):(2 * m), 1:n] = A
    A_mat[(m + 1):(2 * m), (2 * n + 1):(2 * n + m)] = 1 * Matrix(I, m, m)
    A_mat[(2 * m + 1):(2 * m + n), 1:n] = -1 * Matrix(I, n, n)
    A_mat[(2 * m + n + 1):(2 * m + 2 * n), 1:n] = 1 * Matrix(I, n, n)
    A_mat[(2 * m + 1):(2 * m + n), (n + 1):(2 * n)] = 1 * Matrix(I, n, n)
    A_mat[(2 * m + n + 1):(2 * m + 2 * n), (n + 1):(2 * n)] = 1 * Matrix(I, n, n)
    A_mat[2 * (m + n) + 1, (2 * n + 1):(2 * n + m)] = ones(m)
    zero_index_mat = zeros(n, n)
    for index in zero_indices
        zero_index_mat[index, index] = 1
    end
    A_mat[(2 * (m + n) + 2):end, 1:n] = zero_index_mat
    A_mat = sparse(A_mat)

    u_bound = zeros(2 * m + 3 * n + 1)
    u_bound[1:m] = b
    u_bound[(m + 1):(2 * m + 2 * n)] .= Inf
    u_bound[2 * (m + n) + 1] = epsilon
    u_bound[(2 * (m + n) + 2):end] .= 0

    l_bound = zeros(2 * m + 3 * n + 1)
    l_bound[1:m] .= -Inf
    l_bound[(m + 1):(2 * m)] = b
    l_bound[(2 * m + 1):end] .= 0

    model = OSQP.Model()
    OSQP.setup!(model; P=P, q=q, A=A_mat, l=l_bound, u=u_bound)
    results = OSQP.solve!(model)

    opt_x = results.x[1:n]
    opt_val = results.info.obj_val

    return opt_x, opt_val + size(one_indices)[1]

end;

function solveSubproblemDualL1OSQP(A, b, epsilon, gamma;
    solver_output=0, round_solution=true, zero_indices=[], one_indices=[])

    (m, n) = size(A)

    unique_indices = unique([zero_indices; one_indices])
    num_zeros = size(zero_indices)[1]
    num_ones = size(one_indices)[1]
    free_indices = collect(1:n)[(!in).(collect(1:n), Ref(unique_indices))]

    @assert num_ones + num_zeros <= n
    @assert size(unique_indices)[1] == num_ones + num_zeros

    P = zeros(m + 1, m + 1)
    P[1:m, 1:m] = gamma * sum(A[:, i] * A[:, i]' for i in one_indices) / 4
    P = sparse(2 * P)
    q = zeros(m + 1)
    q[1:m] = b
    q[m+1] = epsilon
    A_mat = zeros(n + 2 * m, m + 1)
    A_mat[1:n, 1:m] = A'
    A_mat[(n + 1):(n + m), 1:m] = -1 * Matrix(I, m, m)
    A_mat[(n + 1):(n + m), m + 1] .= 1
    A_mat[(n + m + 1):(n + 2 * m), 1:m] = 1 * Matrix(I, m, m)
    A_mat[(n + m + 1):(n + 2 * m), m + 1] .= 1
    A_mat = sparse(A_mat)

    u_bound = zeros(n + 2 * m)
    l_bound = zeros(n + 2 * m)
    for index in unique_indices
        u_bound[index] = Inf
        l_bound[index] = -Inf
    end
    for index in free_indices
        u_bound[index] = 2 / sqrt(gamma)
        l_bound[index] = -2 / sqrt(gamma)
    end

    u_bound[(n + 1):end] .= Inf
    l_bound[(n + 1):end] .= 0

    model = OSQP.Model()
    OSQP.setup!(model; P=P, q=q, A=A_mat, l=l_bound, u=u_bound, polish=true)
    results = OSQP.solve!(model)

    opt_x = results.x[1:n]
    opt_val = results.info.obj_val

    return opt_x, -opt_val + size(one_indices)[1]

end;
