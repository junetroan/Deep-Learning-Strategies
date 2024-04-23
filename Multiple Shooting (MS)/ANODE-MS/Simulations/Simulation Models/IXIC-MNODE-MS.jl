
# GENERATING DATA TO TEST
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots
using DiffEqFlux
using Distributions
using CSV, Tables, DataFrames
using Random
using StatsBase
gr()

# Collecting Data
data_path = "Multiple Shooting (MS)/ANODE-MS/Data/^IXIC.csv"
data = CSV.read(data_path, DataFrame)

#Train/test Splits

# Data Cleaning and Normalization
date = data[:, 1]
open_price = data[:, 2]

split_ratio = 0.25
train = open_price[1:Int(round(split_ratio*size(open_price, 1))), :]
test = open_price[Int(round(split_ratio*size(open_price, 1))):end, :]

t_train = Float32.(collect(1:Int(round(split_ratio*size(open_price, 1)))))
t_test = Float32.(collect(Int(round(split_ratio*size(open_price, 1))):size(open_price, 1)))

transformer = fit(ZScoreTransform, open_price)
X_train = vec(Float32.(StatsBase.transform(transformer, train)))
X_test = vec(Float32.(StatsBase.transform(transformer, test)))

plot(X_train)

t = collect(1:size(data, 1))
t = Float32.(t)
tspan = (minimum(t_train), maximum(t_train))
tsteps = range(tspan[1], tspan[2], length = length(X_train))

datasize = size(X_train, 1)

# Define the experimental parameter
groupsize = 5
predsize = 5
state = 2

fulltraj_losses = Float32[]

# NUMBER OF ITERATIONS OF THE SIMULATION
iters = 2

i = 2
# Random numbers
rng_1 = StableRNG(i)
rng_2 = StableRNG(i + 2)

# NEURAL NETWORK
U = Lux.Chain(Lux.Dense(state, 30, tanh),
Lux.Dense(30, state))

tsteps = 1.0f0
p, st = Lux.setup(rng_1, U)
params = ComponentVector{Float32}(vector_field_model = p)
u0 = vcat(X_train[1], randn(rng_2, Float32, state - 1))
neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, ComponentArray(p))

tsteps = range(tspan[1], tspan[2], length = length(X_train))

##############################################################################################################################################################
# Testing to reveal variable types and variable content

datasize
groupsize
X_train
prob_node

##############################################################################################################################################################

function group_ranges(datasize::Integer, groupsize::Integer)
    2 <= groupsize <= datasize || throw(DomainError(groupsize,
    "datasize must be positive and groupsize must to be within [2, datasize]"))
    return [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
end

ranges = group_ranges(datasize, groupsize)
u0 = Float32(X_train[first(1:5)])
u0_init = [[X_train[first(rg)]; fill(mean(X_train[rg]), state - 1)] for rg in ranges] 
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
        solver, saveat = tsteps[rg],sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))) 
        for (index, rg) in enumerate(ranges)]

    group_predictions = Array.(sols)


    loss = 0

    for (i, rg) in enumerate(ranges)
        u = X_train[rg] # TODO: make it generic for observed states > 1
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

plot(X_train)
plot!(tester_pred_ss[2, :])

function loss_single_shooting(p)
    pred = predict_single_shooting(p)
    l = loss_function(X_train, pred[1,:])
    return l, pred
end

ls, ps = loss_single_shooting(params.vector_field_model)

continuity_term = 10.0
params = ComponentVector{Float32}(θ = p, u0_init = u0_init)
ls, ps = multiple_shoot_mod(params, X_train, tsteps, prob_node, loss_function, continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize; continuity_term)

# Modified multiple_shoot method 
multiple_shoot_mod(params, X_train, tsteps, prob_node, loss_function,
    continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize;
    continuity_term)


function loss_multiple_shoot(p)
    return multiple_shoot_mod(p, X_train, tsteps, prob_node, loss_function,
        continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), groupsize;
        continuity_term)[1]
end

test_multiple_shoot = loss_multiple_shoot(params)

loss_single_shooting(params.θ)[1]

losses = Float32[]

callback = function (θ, l)
    push!(losses, loss_multiple_shoot(θ))
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")

    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shoot(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(),  maxiters = 5000, callback = callback) 


losses_df = DataFrame(loss = losses)
CSV.write("Multiple Shooting (MS)/ANODE-MS/Simulations/Results/sim-LV-MNODE-MS/Loss Data/Losses $i.csv", losses_df, writeheader = false)

loss_ms, _ = loss_single_shooting(res_ms.u.θ)
preds = predict_single_shooting(res_ms.u.θ)

plot(preds[1, :])
scatter!(X_train)

gr()
function plot_results(real, pred, t)
    plot(t, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised MNODE-MS Model", xlabel="Time", ylabel="Population")
    scatter!(t, real, label = "Training Data")
    plot!(legend=:topright)
    savefig("Multiple Shooting (MS)/ANODE-MS/Simulations/Results/sim-LV-MNODE-MS/Plots/Simulation $i.png")
end

plot_results(X_train, preds, t_train)




