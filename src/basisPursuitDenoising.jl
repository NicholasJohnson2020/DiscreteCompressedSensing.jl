include("helperLibrary.jl")

function basisPursuitDenoising(A, b, epsilon;
    norm_function="L2", solver_output=0, solver="Gurobi", round_solution=true)

    @assert solver in ["Gurobi", "SCS"]
    @assert norm_function in ["L2", "L1"]

    (m, n) = size(A)

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

    if norm_function == "L2"
        @constraint(model, sum(abs_residual[i]^2 for i=1:m) <= epsilon)
    else
        @constraint(model, sum(abs_residual[i] for i=1:m) <= epsilon)
    end

    @objective(model, Min, sum(abs_x[i] for i=1:n))

    optimize!(model)

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
