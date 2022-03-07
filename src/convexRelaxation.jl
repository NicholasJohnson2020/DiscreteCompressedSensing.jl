function perspectiveRelaxation(A, b, epsilon, lambda;
    solver_output=0, solver="Gurobi", round_solution=true)

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

    if round_solution
        rounded_x, num_support = roundSolution(opt_z, A, b, epsilon)
        return num_support, rounded_x, opt_x, opt_z, obj_value
    else
        return opt_x, opt_z, obj_value
    end

end;


function perspectiveFormulation(A, b, epsilon, lambda; solver_output=0, solver="Gurobi")

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
    @variable(model, z[i=1:n], Bin)
    @variable(model, theta[i=1:n] >= 0)
    @variable(model, residual[i=1:m])

    @constraint(model, residual .== A * x .- b)
    @constraint(model, sum(residual[i]^2 for i=1:m) <= epsilon)

    @constraint(model, [i=1:n], z[i] * theta[i] >= x[i]^2)

    @objective(model, Min, sum(z[i] + theta[i] / lambda for i=1:n))

    optimize!(model)

    obj_value = objective_value(model)
    opt_x = value.(x)
    opt_z = value.(z)

    return opt_x, opt_z, obj_value

end;
