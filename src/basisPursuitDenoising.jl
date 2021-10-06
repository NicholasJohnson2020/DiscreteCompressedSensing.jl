function basisPursuitDenoising(A, b, epsilon; solver_output=0, solver="Gurobi")

    @assert solver in ["Gurobi", "SCS"]

    (m, n) = size(A)

    if solver == "Gurobi"
        model = Model(with_optimizer(Gurobi.Optimizer))
        set_optimizer_attribute(model, "OutputFlag", solver_output)
    else
        model = Model(SCS.Optimizer)
        set_optimizer_attribute(model, "verbose", solver_output)
    end

    @variable(model, x[i=1:n])
    @variable(model, abs_x[i=1:n]>=0)
    @variable(model, residual[i=1:m])

    @constraint(model, [i=1:n], abs_x[i] >= x[i])
    @constraint(model, [i=1:n], -abs_x[i] <= x[i])
    @constraint(model, residual .== A * x .- b)

    @constraint(model, sum(residual[i]^2 for i=1:m) <= epsilon)

    @objective(model, Min, sum(x[i] for i=1:n))

    optimize!(model)

    opt_x = value.(x)

    return opt_x

end;
