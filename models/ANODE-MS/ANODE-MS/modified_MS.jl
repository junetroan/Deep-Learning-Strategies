# Created on Monday 8th of April by collaboration of Vinicius Santana Viena and June Mari Berge Trøan

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
@time solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 1.0)
X = Array(solution)
x = X[1,:]
#x2 = zeros(length(x))
#X = [x x2]'
# SPECIFICATIONS
group_size = 5
state = 2
groupsize = 5
predsize = 5
tsteps = 1

# NEURAL NETWORK
U = Lux.Chain(Lux.Dense(state, 30, tanh),
              Lux.Dense(30, state))
              
p, st = Lux.setup(rng, U)
params = ComponentVector{Float32}(vector_field_model = p)
neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, ComponentArray(p))


datasize = size(x, 1)
tspan = (0.0f0, 10.0f0) # Original (0.0f0, 10.0f0
tsteps = range(tspan[1], tspan[2]; length = datasize)
solve(prob_node, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps)

##############################################################################################################################################################

function group_ranges(datasize::Integer, groupsize::Integer)
    2 <= groupsize <= datasize || throw(DomainError(groupsize,
        "datasize must be positive and groupsize must to be within [2, datasize]"))
    return [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
end

ranges = group_ranges(datasize, group_size)
u0 = Float32(x[first(1:5)])
u0_init = [[x[first(rg)]; fill(mean(x[rg]), state - 1)] for rg in ranges] 
u0_init = mapreduce(permutedims, vcat, u0_init)

function multiple_shoot_mod(p, ode_data, tsteps, prob::ODEProblem, loss_function::F,
    continuity_loss::C, solver::SciMLBase.AbstractODEAlgorithm, group_size::Integer;
    continuity_term::Real = 100, kwargs...) where {F, C}

    datasize = size(ode_data, 1)

    if group_size < 2 || group_size > datasize
        throw(DomainError(group_size, "group_size can't be < 2 or > number of data points"))
    end

    ranges = group_ranges(datasize, group_size)

    sols = [solve(remake(prob_node; p = p.θ, tspan = (tsteps[first(rg)], tsteps[last(rg)]),
        u0 = p.u0_init[index, :]),
       solver; saveat = tsteps[rg]) 
        for (index, rg) in enumerate(ranges)]

    group_predictions = Array.(sols)


    loss = 0

    for (i, rg) in enumerate(ranges)
        u = x[rg] # TODO: make it generic for observed states > 1
        û = group_predictions[i][1, :]
        loss += loss_function(u, û)

        if i > 1
            # Ensure continuity between last state in previous prediction
            # and current initial condition in ode_data
            loss += continuity_term *
                    continuity_loss(group_predictions[i - 1][:, end], group_predictions[i][:, 1])
        end
    end

    return loss, group_predictions
end

#Testing
# Calculate multiple shooting loss
loss_function(data, pred) = sum(abs2, data - pred)
continuity_loss(uᵢ₊₁, uᵢ) = sum(abs2, uᵢ₊₁ - uᵢ)
predict_single_shooting(p) = Array(first(neuralode(u0_init[1,:],p,st)))
tester_pred_ss = predict_single_shooting(params.vector_field_model)

function loss_single_shooting(p)
    pred = predict_single_shooting(p)
    l = loss_function(x, pred[1,:])
    return l, pred
end

ls, ps = loss_single_shooting(params.vector_field_model)

continuity_term = 10.0
params = ComponentVector{Float32}(θ = p, u0_init = u0_init)
ls, ps = multiple_shoot_mod(params, x, tsteps, prob_node, loss_function, continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size; continuity_term)

#lossses, preds = predict_single_shooting(params.p)

# Modified multiple_shoot method 
multiple_shoot_mod(params, x, tsteps, prob_node, loss_function,
    continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size;
    continuity_term)

function loss_multiple_shoot(p)
    return multiple_shoot_mod(p, x, tsteps, prob_node, loss_function,
        continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size;
        continuity_term)
end

test_multiple_shoot = loss_multiple_shoot(params)
test_multiple_shoot[1]

losses = Float32[]

callback = function (θ, l)
    push!(losses, loss_single_shooting(θ)[1])
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")

    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shoot(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(),  maxiters = 5000) #callback  = callback,

loss_ms, _ = loss_single_shooting(res_ms.u.θ)
preds = predict_single_shooting(res_ms.u.θ)


plot(preds[1, :])
scatter!(x)