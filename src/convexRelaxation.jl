function SOSRelaxation(A, b, epsilon, lambda; solver_output=false,
    use_default_lambda=false, relaxation_degree=1)
    """
    This function computes the solution to problem (21) of the accompanying
    paper.

    :param A: A m-by-n design matrix.
    :param b: An m dimensional vector of observations.
    :param epsilon: A numerical threshold parameter (Float64).
    :param lambda: A regularization parameter (Float64).
    :param solver_output: Flag that determines Mosek is called in verbose
                          mode (Bool).
    :param use_default_lambda: Flag that determines whether or not to use the
                               default value of the regularization paramter
                               (Bool).
    :param relaxation_degree: The degree of the relaxation (Int64).

    :return: This function returns the optimal value of problem (21).
    """
    (m, n) = size(A)

    if use_default_lambda
        lambda = sqrt(n)
    end

    @polyvar x[1:n] z[1:n]
    obj_func = sum(z[i] + x[i]^2/lambda for i=1:n)
    basis_large = monomials([x; z], 0:relaxation_degree)
    basis_small = monomials([x; z], 0:(relaxation_degree-1))

    # Build the optimization problem
    model = Model(Mosek.Optimizer)
    set_optimizer_attribute(model, MOI.Silent(), !solver_output)

    @variable(model, t[i=1:n], Poly(basis_small))
    @variable(model, s_0, SOSPoly(basis_large))
    @variable(model, s_1, SOSPoly(basis_small))
    @variable(model, opt_val)

    term_1 = sum(t[i] * (z[i]*x[i]-x[i]) for i=1:n)
    term_2 = sum((z[i]^2-z[i]) for i=1:n)
    term_3 = s_1 * (epsilon-(A*x-b)'*(A*x-b))
    @constraint(model, obj_func - opt_val == term_1 - term_2 + s_0 + term_3)

    @objective(model, Max, opt_val)

    optimize!(model)

    @assert termination_status(model) == MOI.OPTIMAL
    return objective_value(model)

end

function perspectiveRelaxation(A, b, epsilon, lambda;
    solver_output=0, round_solution=true,
    use_default_lambda=false)
    """
    This function computes the solution to problem (13) of the accompanying
    paper.

    :param A: A m-by-n design matrix.
    :param b: An m dimensional vector of observations.
    :param epsilon: A numerical threshold parameter (Float64).
    :param lambda: A regularization parameter (Float64).
    :param solver_output: The "OutputFlag" parameter to be passed to
                          Gurobi (Int64).
    :param round_solution: Flag that controls whether or not to perform a greedy
                           rounding of the solution of (13) to obtain a vector
                           that is feasible to (3) (Bool).
    :param use_default_lambda: Flag that determines whether or not to use the
                              default value of the regularization paramter
                              (Bool).

    :return: If round_solution is true, this function returns six values:
             1) The cardinality of the rounded solution (Int64).
             2) The rounded solution.
             3) The x vector solution to (13).
             4) The z vector solution to (13).
             5) The optimal value of problem (13).
             6) The amount of time in milliseconds to perform the rounding.

             If round_solution is false, this function returns three values:
             1) The x vector solution to (13).
             2) The z vector solution to (13).
             3) The optimal value of problem (13).
    """
    (m, n) = size(A)

    if use_default_lambda
        lambda = sqrt(n)
    end

    # Build the optimization problem
    model = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, x[i=1:n])
    @variable(model, z[i=1:n] >= 0)
    @variable(model, theta[i=1:n] >= 0)
    @variable(model, residual[i=1:m])

    @constraint(model, residual .== A * x .- b)
    @constraint(model, sum(residual[i]^2 for i=1:m) <= epsilon)

    @constraint(model, [i=1:n], z[i] * theta[i] >= x[i]^2)

    @constraint(model, [i=1:n], z[i] <= 1)

    @objective(model, Min, sum(z[i] + theta[i] / lambda for i=1:n))

    optimize!(model)

    obj_value = objective_value(model)
    opt_x = value.(x)
    opt_z = value.(z)

    # Perform greedy rounding
    if round_solution
        rounding_start = now()
        rounded_x, num_support = roundSolution(opt_x, A, b, epsilon)
        rounding_end = now()
        rounding_time = rounding_end - rounding_start

        return (num_support,
                rounded_x,
                opt_x, opt_z, obj_value,
                rounding_time)
    else
        return opt_x, opt_z, obj_value
    end

end;


function perspectiveFormulation(A, b, epsilon, lambda; solver_output=0,
    BPD_backbone=false, use_default_lambda=false)
    """
    This function computes the solution to problem (12) of the accompanying
    paper by directly calling Gurobi.

    :param A: A m-by-n design matrix.
    :param b: An m dimensional vector of observations.
    :param epsilon: A numerical threshold parameter (Float64).
    :param lambda: A regularization parameter (Float64).
    :param solver_output: The "OutputFlag" parameter to be passed to
                          Gurobi (Int64).
    :param BPD_backbone: Flag that controls whether or not to use the BPD
                         solution as a backbone (Bool).
    :param use_default_lambda: Flag that determines whether or not to use the
                              default value of the regularization paramter
                              (Bool).

    :return: This function returns three values:
             1) The x vector solution to (12).
             2) The z vector solution to (12).
             3) The optimal value of problem (12).
    """
    (m, n) = size(A)
    original_n = n
    backbone = []

    # Compute the BPD backbone if necessary
    if BPD_backbone

        _, opt_x = basisPursuitDenoising(A, b, epsilon,
                                         round_solution=false)

        for index=1:size(opt_x)[1]
            if abs(opt_x[index]) > 1e-6
                append!(backbone, index)
            end
        end
        n = size(backbone)[1]
        reduced_A = zeros(m, n)
        for i=1:n
            reduced_A[:, i] = A[:, backbone[i]]
        end
        A = reduced_A
    end

    if use_default_lambda
        lambda = sqrt(n)
    end

    # Build the optimization problem
    model = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, x[i=1:n])
    @variable(model, z[i=1:n], Bin)
    @variable(model, theta[i=1:n] >= 0)
    @variable(model, abs_residual[i=1:m])

    @constraint(model, abs_residual .>= A * x .- b)
    @constraint(model, abs_residual .>= -A * x .+ b)

    @constraint(model, sum(abs_residual[i]^2 for i=1:m) <= epsilon)

    @constraint(model, [i=1:n], z[i] * theta[i] >= x[i]^2)

    @objective(model, Min, sum(z[i] + theta[i] / lambda for i=1:n))

    optimize!(model)

    println(termination_status(model))
    obj_value = objective_value(model)
    opt_x = value.(x)
    opt_z = value.(z)

    if BPD_backbone
        temp_x = zeros(original_n)
        temp_z = zeros(original_n)
        for i=1:n
            temp_x[backbone[i]] = opt_x[i]
            temp_z[backbone[i]] = opt_z[i]
        end
        opt_x = temp_x
        opt_z = temp_z
    end

    return opt_x, opt_z, obj_value

end;
