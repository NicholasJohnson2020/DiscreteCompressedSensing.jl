include("helperLibrary.jl")

function basisPursuitDenoising(A, b, epsilon; weights=nothing,
    solver_output=0, solver="Gurobi", round_solution=true, max_weight=1e6)

    @assert solver in ["Gurobi", "SCS"]

    (m, n) = size(A)

    if weights == nothing
        weights = ones(n)
    end

    @assert size(weights)[1] == n

    active_indices = findall(<=(max_weight), weights)
    zero_indices = findall(>(max_weight), weights)

    if solver == "Gurobi"
        model = Model(with_optimizer(Gurobi.Optimizer, GUROBI_ENV))
        set_optimizer_attribute(model, "OutputFlag", solver_output)
    else
        model = Model(SCS.Optimizer)
        set_optimizer_attribute(model, "verbose", solver_output)
    end

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

    if round_solution
        rounding_start = now()
        rounded_x, num_support = roundSolution(opt_x, A, b, epsilon,
                                               norm_function=norm_function)
        rounding_end = now()
        rounding_time = rounding_end - rounding_start
        return num_support, rounded_x, sum(abs.(opt_x) .> 1e-6), opt_x, rounding_time
    else
        return sum(abs.(opt_x) .> 1e-6), opt_x
    end

end;

function iterativeReweightedL1(A, b, epsilon; solver_output=0, solver="Gurobi",
    round_solution=true, max_iter=50, numerical_stability_param=1e-6)

    current_card, current_x = basisPursuitDenoising(A, b, epsilon,
                                                    solver_output=solver_output,
                                                    solver=solver,
                                                    round_solution=false)
    iter_count = 0
    while iter_count < max_iter
        new_weights = 1 ./ abs.(current_x) .+ numerical_stability_param
        output = basisPursuitDenoising(A, b, epsilon,
                                       solver_output=solver_output,
                                       solver=solver,
                                       weights=new_weights,
                                       round_solution=false)
        if output == nothing
            break
        end
        new_card, new_x = output
        println(new_card)
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
