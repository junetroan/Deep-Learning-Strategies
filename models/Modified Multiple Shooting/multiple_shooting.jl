"""
    multiple_shoot(p, ode_data, tsteps, prob, loss_function,
        [continuity_loss = _default_continuity_loss], solver, group_size;
        continuity_term = 100, kwargs...)

Returns a total loss after trying a 'Direct multiple shooting' on ODE data and an array of
predictions from each of the groups (smaller intervals). In Direct Multiple Shooting, the
Neural Network divides the interval into smaller intervals and solves for them separately.
The default continuity term is 100, implying any losses arising from the non-continuity
of 2 different groups will be scaled by 100.

Arguments:

  - `p`: The parameters of the Neural Network to be trained.
  - `ode_data`: Original Data to be modelled.
  - `tsteps`: Timesteps on which ode_data was calculated.
  - `prob`: ODE problem that the Neural Network attempts to solve.
  - `loss_function`: Any arbitrary function to calculate loss.
  - `continuity_loss`: Function that takes states ``\\hat{u}_{end}`` of group ``k`` and
    ``u_{0}`` of group ``k+1`` as input and calculates prediction continuity loss between
    them. If no custom `continuity_loss` is specified, `sum(abs, û_end - u_0)` is used.
  - `solver`: ODE Solver algorithm.
  - `group_size`: The group size achieved after splitting the ode_data into equal sizes.
  - `continuity_term`: Weight term to ensure continuity of predictions throughout
    different groups.
  - `kwargs`: Additional arguments splatted to the ODE solver. Refer to the
    [Local Sensitivity Analysis](https://docs.sciml.ai/SciMLSensitivity/stable/) and
    [Common Solver Arguments](https://docs.sciml.ai/DiffEqDocs/stable/basics/common_solver_opts/)
    documentation for more details.

!!! note

    The parameter 'continuity_term' should be a relatively big number to enforce a large penalty
    whenever the last point of any group doesn't coincide with the first point of next group.
"""

# GENERATING DATA TO TEST
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots
using DiffEqFlux
using Plots
using Distributions

gr()
rng = StableRNG(1111)
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

tspan = (0.0f0, 10.0f0)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
println("solving ODE")
@time solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat =1)
X = Array(solution)
x = X[1,:]
#x2 = zeros(length(x))
#X = [x x2]'
# SPECIFICATIONS
group_size = 5
state = 1
groupsize = 5
predsize = 5
tsteps = 1

# NEURAL NETWORK
U = Lux.Chain(Lux.Dense(state, 30, tanh),
              Lux.Dense(30, state))
p, st = Lux.setup(rng, U)
params = ComponentVector{Float32}(vector_field_model = p)
u0 = 5.0f0 * rand(rng, Float32)
neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> U(u)[1], u0, tspan, ComponentArray(p))



datasize = size(x,1)
tspan = (0.0f0, 10.0f0) # Original (0.0f0, 10.0f0
tsteps = range(tspan[1], tspan[2]; length = datasize)

##############################################################################################################################################################
function group_ranges(datasize::Integer, groupsize::Integer)
    2 <= groupsize <= datasize || throw(DomainError(groupsize,
        "datasize must be positive and groupsize must to be within [2, datasize]"))
    return [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
end

if group_size < 2 || group_size > datasize
    throw(DomainError(group_size, "group_size can't be < 2 or > number of data points"))
end

ranges = group_ranges(datasize, group_size)
u0 = Float32(x[first(1:5)])

#sols = [solve(remake(prob_node; tspan = (tsteps[first(rg)], tsteps[last(rg)]), u0 = x[first(rg)]), AutoTsit5(Rosenbrock23(autodiff=false)); saveat = tsteps) for rg in ranges]
sols = [solve(remake(prob_node; p, tspan = (tsteps[first(rg)], tsteps[last(rg)]), u0 = x[first(rg)]), AutoTsit5(Rosenbrock23(autodiff=false)); saveat = tsteps[rg]) for rg in ranges]

group_predictions = Array.(sols)

test_in = [1.0f0]
test_out = U(test_in, p, st)[1]
println("Test input: ", test_in)
println("Test output: ", test_out)

#=
n_new = [remake(prob_node; p, tspan = (tsteps[first(rg)], tsteps[last(rg)]), u0 = x[first(rg)]) for rg in ranges]
sensealg = ReverseDiffAdjoint()
solutions = [solve(n_new, Tsit5(), saveat = tsteps[rg]) for rg in ranges]

Array(solve(shooting_problem, verbose = false,  AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1f-6, reltol = 1f-6, 
p=θ, trajectories = length(u0_vec), sensealg = sensealg))


group_predictions = Array.(solutions)


function multiple_shoot(p, ode_data, tsteps, prob::ODEProblem, loss_function::F,
        continuity_loss::C, solver::SciMLBase.AbstractODEAlgorithm, group_size::Integer;
        continuity_term::Real = 100, kwargs...) where {F, C}

    datasize = size(x, 1) #Changed from 2 to 1
    if group_size < 2 || group_size > datasize
        throw(DomainError(group_size, "group_size can't be < 2 or > number of data points"))
    end

    # Get ranges that partition data to groups of size group_size
    ranges = group_ranges(datasize, group_size)

    # Multiple shooting predictions
    sols = [solve(
                remake(prob; p, tspan = (tsteps[first(rg)], tsteps[last(rg)]),
                    u0 = ode_data[:, first(rg)]),
                solver;
                saveat = tsteps[rg],
                kwargs...)
            for rg in ranges]

    group_predictions = Array.(sols)

    # Abort and return infinite loss if one of the integrations failed
    retcodes = [sol.retcode for sol in sols]
    all(SciMLBase.successful_retcode, retcodes) || return Inf, group_predictions

    # Calculate multiple shooting loss
    loss = 0
    for (i, rg) in enumerate(ranges)
        u = ode_data[:, rg]
        û = group_predictions[i]
        loss += loss_function(u, û)

        if i > 1
            # Ensure continuity between last state in previous prediction
            # and current initial condition in ode_data
            loss += continuity_term *
                    continuity_loss(group_predictions[i - 1][:, end], u[:, 1])
        end
    end

    return loss, group_predictions
end

function multiple_shoot(p, ode_data, tsteps, prob::ODEProblem, loss_function::F,
        solver::SciMLBase.AbstractODEAlgorithm, group_size::Integer; kwargs...) where {F}
    return multiple_shoot(p, ode_data, tsteps, prob, loss_function,
        _default_continuity_loss, solver, group_size; kwargs...)
end

=#