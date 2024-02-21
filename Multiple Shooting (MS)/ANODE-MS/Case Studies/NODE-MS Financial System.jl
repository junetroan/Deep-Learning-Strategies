#NODE-MS Financial System

# Test againt Julias multiple shooting
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Distributions
using ComponentArrays, Lux, DiffEqFlux, Optimization, OptimizationPolyalgorithms, DifferentialEquations, Plots
using DiffEqFlux: group_ranges
using OptimizationOptimisers
using StableRNGs
gr()

# Collecting data
data_path = "Multiple Shooting (MS)/ANODE-MS/Case Studies/financial_time_series.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits

split_ration = 0.75
train = data[1:Int(round(split_ration*size(data, 1))), 2]
test = data[Int(round(split_ration*size(data, 1))):end, 2]

# Data Cleaning and Normalization
t = data[:,1]
ir = data[:,2]

transformer = fit(ZScoreTransform, ir)
X_train_ref = StatsBase.transform(transformer, train)#[:, 1])
X_train = X_train_ref[1:1500]
t_train = data[1:1500, 1]

X_test = StatsBase.transform(transformer, test[:, 1])
t_test = collect(Int(round(split_ration*size(data, 1))):size(data, 1))

rng = StableRNG(1111)
group_size = 5
state = 2
continuity_term = 10.0f0
tspan = (minimum(t_train), maximum(t_train))

# Define the Neural Network
nn = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))
p_init, st = Lux.setup(rng, nn)
u0 = X_train[1]

neuralode = NeuralODE(nn, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> nn(u,p,st)[1], [[u0[1];zeros(state-1)]], tspan, ComponentArray(p_init))

# Define parameters for Multiple Shooting
function loss_function(data, pred)
	return sum(abs2, data - pred)
end

function continuity_loss_function(u_end, u_0)
    return mean(abs2, u_end - u_0)
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, X, tsteps, prob_node, loss_function, AutoTsit5(Rosenbrock23(autodiff=false)),
                          group_size; continuity_term)
end


function predict_final(θ)
    return Array(neuralode([u0[1]; zeros(state -1)], θ, st)[1])
end

function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, X_train .- X̂[1, :])
    prediction_error
end


function predict_single_shooting(p)
    return Array(neuralode([[u0[1];zeros(state-1)]], p, st)[1])
end


losses = Float32[]

callback = function (p, l, preds; doplot = false)
    push!(losses, final_loss(p))
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")

    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters=5000; callback = callback)

full_traj = predict_final(res_ms.u)
full_traj_loss = final_loss(res_ms.u)
plot(full_traj[1, :])
scatter!(X[1, :])