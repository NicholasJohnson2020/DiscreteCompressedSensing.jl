include("helperLibrary.jl")

function basisPursuitDenoising(A, b, epsilon; weights=nothing,
    solver_output=0, round_solution=true, max_weight=1e6)
    """
    This function computes the solution to problem (5) and (7) of the
    accompanying paper.

    :param A: A m-by-n design matrix.
    :param b: An m dimensional vector of observations.
    :param epsilon: A numerical threshold parameter (Float64).
    :param weights: A n dimensional vector of weights.
    :param solver_output: The "OutputFlag" parameter to be passed to
                          Gurobi (Int64).
    :param round_solution: Flag that controls whether or not to perform a greedy
                           rounding of the solution of (5) to further sparsify.
    :param max_weight: Maximum allowable weight (Float64).

    :return: If round_solution is true, this function returns five values:
             1) The cardinality of the rounded solution (Int64).
             2) The rounded solution.
             3) The cardinality of the solution to (5) (Int64).
             4) The x vector solution to (5).
             5) The amount of time in milliseconds to perform the rounding.

             If round_solution is false, this function returns two values:
             1) The cardinality of the solution to (5) (Int64).
             2) The x vector solution to (5).
    """
    (m, n) = size(A)

    if weights == nothing
        weights = ones(n)
    end

    @assert size(weights)[1] == n

    active_indices = findall(<=(max_weight), weights)
    zero_indices = findall(>(max_weight), weights)

    # Build the optimization problem
    model = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
    set_optimizer_attribute(model, "OutputFlag", solver_output)

    @variable(model, x[i=1:n])
    @variable(model, abs_x[i=1:n]>=0)
    @variable(model, abs_residual[i=1:m])

    @constraint(model, [i=1:n], abs_x[i] >= x[i])
    @constraint(model, [i=1:n], -abs_x[i] <= x[i])
    @constraint(model, abs_residual .>= A * x .- b)
    @constraint(model, abs_residual .>= -A * x .+ b)
    @constraint(model, [i in zero_indices], x[i] == 0)

    @constraint(model, sum(abs_residual[i]^2 for i=1:m) <= epsilon)

    @objective(model, Min, sum(weights[i] * abs_x[i] for i in active_indices))

    optimize!(model)

    if termination_status(model) != MOI.OPTIMAL
        return nothing
    end

    opt_x = value.(x)

    # Perform greedy rounding if necessary
    if round_solution
        rounding_start = now()
        rounded_x, num_support = roundSolution(opt_x, A, b, epsilon)
        rounding_end = now()
        rounding_time = rounding_end - rounding_start
        return num_support, rounded_x, sum(abs.(opt_x) .> 1e-6), opt_x, rounding_time
    else
        return sum(abs.(opt_x) .> 1e-6), opt_x
    end

end;

function iterativeReweightedL1(A, b, epsilon; solver_output=0,
    round_solution=true, max_iter=50, numerical_stability_param=1e-6)
    """
    This function performs iterated reweighted L1 minimization for problem (7)
    defined in Section 2.2 of the accompanying paper.

    :param A: A m-by-n design matrix.
    :param b: An m dimensional vector of observations.
    :param epsilon: A numerical threshold parameter (Float64).
    :param solver_output: The "OutputFlag" parameter to be passed to
                          Gurobi (Int64).
    :param round_solution: Flag that controls whether or not to perform a greedy
                           rounding of the solution of (7) to further sparsify.
    :param max_iter: Maximum allowable reweighting iterations (Int64).
    :param numerical_stability_param: Stability paramter for the iterated
                                      reweighted L1 algorithm (Float64).

    :return: If round_solution is true, this function returns six values:
             1) The cardinality of the rounded solution (Int64).
             2) The rounded solution.
             3) The cardinality of the solution to (7) (Int64).
             4) The x vector solution to (7).
             5) The amount of time in milliseconds to perform the rounding.
             6) The number of reweighting iterations (Int64).

             If round_solution is false, this function returns three values:
             1) The cardinality of the solution to (7) (Int64).
             2) The x vector solution to (7).
             3) The number of reweighting iterations (Int64).
    """
    # Solve (7) with uniform weights to initiliaze
    current_card, current_x = basisPursuitDenoising(A, b, epsilon,
                                                    solver_output=solver_output,
                                                    round_solution=false)
    iter_count = 0
    # Main loop
    while iter_count < max_iter
        new_weights = 1 ./ abs.(current_x) .+ numerical_stability_param
        # Solve (7) with newly computed weights
        output = basisPursuitDenoising(A, b, epsilon,
                                       solver_output=solver_output,
                                       weights=new_weights,
                                       round_solution=false)
        if output == nothing
            break
        end
        new_card, new_x = output
        if new_card > current_card
            break
        end
        if norm(new_x-current_x) <= 1e-6
            break
        end
        current_x = new_x
        current_card = new_card
        iter_count = iter_count + 1
    end

    # Perform greedy rounding if necessary
    if round_solution
        rounding_start = now()
        rounded_x, num_support = roundSolution(current_x, A, b, epsilon)
        rounding_end = now()
        rounding_time = rounding_end - rounding_start
        return num_support, rounded_x,
               sum(abs.(current_x) .> 1e-6), current_x,
               rounding_time, iter_count
    else
        return sum(abs.(current_x) .> 1e-6), current_x, iter_count
    end

end;
