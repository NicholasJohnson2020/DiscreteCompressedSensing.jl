function solveSubproblemPrimal(A, b, epsilon, gamma;
    solver_output=0, round_solution=true, zero_indices=[], one_indices=[])
    """
    This function solves problem (25) Section 5.1.1 of the accompanying paper.

    :param A: A m-by-n design matrix.
    :param b: An m dimensional vector of observations.
    :param epsilon: A numerical threshold parameter (Float64).
    :param gamma: A regularization parameter (Float64).
    :param solver_output: The value of the OutputFlag parameter of the Gurobi
                          solver (Int64).
    :param round_solution: A boolean indicating whether or not to round the
                           solution of the optimization problem.
    :param zero_indices: A list of indices between 1 and n.
    :param one_indices: A list of indices between 1 and n.

    :return: If round_solution is true, this function returns five values. The
             first is the rounded solution vector, the second is the objective
             value achieved by the rounded solution, the third is the x vector
             solution of the optimization problem, the fourth is the z vector
             solution of the optimization problem and the fifth is the optimal
             value of the optimization problem. If round_solution is false,
             this function returns three values. The first is the x vector
             solution of the optimization problem, the second is the z vector
             solution of the optimization problem and the third is the optimal
             value of the optimization problem.
    """
    (m, n) = size(A)

    # Identify the free indices
    unique_indices = unique([zero_indices; one_indices])
    num_zeros = size(zero_indices)[1]
    num_ones = size(one_indices)[1]
    free_indices = collect(1:n)[(!in).(collect(1:n), Ref(unique_indices))]

    @assert num_ones + num_zeros <= n
    @assert size(unique_indices)[1] == num_ones + num_zeros

    # Build the optimization problem
    # (for julia 1.5.2) model = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
    model = Model(()->Gurobi.Optimizer(GUROBI_ENV))
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

    # Round the optimization problem solution if necessary and return the output
    if round_solution
        rounded_x, num_support = roundSolution(opt_x, A, b, epsilon)

        return rounded_x, num_support + norm(rounded_x)^2 / gamma,
               opt_x, opt_z, objective_value(model) + size(one_indices)[1]
    else
        return opt_x, opt_z, objective_value(model) + size(one_indices)[1]
    end

end;

function solveSubproblemDual(A, b, epsilon, gamma;
    solver_output=0, zero_indices=[], one_indices=[])
    """
    This function solves problem (26) Section 5.1.1 of the accompanying paper.

    :param A: A m-by-n design matrix.
    :param b: An m dimensional vector of observations.
    :param epsilon: A numerical threshold parameter (Float64).
    :param gamma: A regularization parameter (Float64).
    :param solver_output: The value of the OutputFlag parameter of the Gurobi
                          solver (Int64).
    :param zero_indices: A list of indices between 1 and n.
    :param one_indices: A list of indices between 1 and n.

    :return: This function returns two values. The first is the optimal
             solution of the optimization problem and the second is the optimal
             value of the optimization problem.
    """
    (m, n) = size(A)

    # Identify the free indices
    all_indices = collect(1:n)
    free_indices = all_indices[(!in).(all_indices, Ref(zero_indices))]
    free_indices = free_indices[(!in).(free_indices, Ref(one_indices))]

    # Build the optimization problem
    # (for julia 1.5.2) model = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
    model = Model(()->Gurobi.Optimizer(GUROBI_ENV))
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
