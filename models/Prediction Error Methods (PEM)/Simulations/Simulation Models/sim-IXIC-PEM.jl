
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots, Random
using CSV, Tables, DataFrames
using DataInterpolations
using OrdinaryDiffEq
using OptimizationPolyalgorithms
using DiffEqFlux
using Plots
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
y = X_train
t = collect(1:size(data, 1))
t = Float32.(t)
tspan = (minimum(t_train), maximum(t_train))
tsteps = range(tspan[1], tspan[2], length = length(X_train))

datasize = size(X_train, 1)

# Interpolation of given data
y_zoh = LinearInterpolation(y, tsteps)

# Definition of neural network
i = 2
state = 2

println("Simulation $i")

rng1 = StableRNG(i+1)
rng2 = StableRNG(i+2)
u0 = vcat(y[1], randn(rng2, Float32, state - 1))
K = rand(rng2, Float32, 2)

U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p, st = Lux.setup(rng1, U)

function predictor!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    yt = y_zoh(t)
    e = yt .- û[1]
    du[1:end] =  û[1:end] .+ abs.(p.K) .* e
end

params = ComponentVector{Float32}(vector_field_model = p, K = K)
prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat = tsteps )
soln_nn = solve(prob_nn, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = tsteps )

#=
t = soln_nn.t  
soln_nn = Array(soln_nn)
=#
        
function prediction(p)     
    _prob = remake(prob_nn, u0 = u0, p = p)     
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))     
    Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = tsteps , sensealg = sensealg)) 
end

function predloss(p)
    yh = prediction(p)
    e2 = mean(abs2, y .- yh[1,:])
    return e2
end

losses = Float32[]
Ks = []

callback = function (θ, l)
    push!(losses, predloss(θ))
    push!(Ks, θ.K[1:end])

    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000, verbose = false, callback=callback)

losses_df = DataFrame(losses = losses)
Ks_mat = mapreduce(permutedims, vcat, Ks)
Ks_df = DataFrame(Ks_mat, :auto)

CSV.write("Prediction Error Methods (PEM)/Simulations/Results/sim-IXIC/Loss Data/Losses.csv", losses_df, writeheader = false)
CSV.write("Prediction Error Methods (PEM)/Simulations/Results/sim-IXIC/Ks data/Ks.csv", Ks_df, writeheader = false)

full_traj = prediction(res_ms.u)
full_traj_loss = predloss(res_ms.u)

function plot_results(t, real, pred)
    plot(t, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised PEM Model", xlabel = "Time", ylabel = "Population")
    plot!(t, real, label = "Training Data")
    plot!(legend=:topleft)
    savefig("Prediction Error Methods (PEM)/Simulations/Results/sim-IXIC/Plots/Simulation.png")
end

plot_results(t_train, y, full_traj)        

        
 