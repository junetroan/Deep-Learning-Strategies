#Test with F1 telemetry data

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
using Statistics
using StatsBase
gr()

data_path = "Multiple Shooting (MS)/ANODE-MS/Data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
split_ration = 0.25
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
t = 0.0:1.0:581
speed = data[:,1]

train_data = convert(Vector{Float32}, train[:,2])
test_data = convert(Vector{Float32}, test[:,2])

transformer = fit(ZScoreTransform, train_data)
X_train = StatsBase.transform(transformer, train_data)
X_test = StatsBase.transform(transformer, test_data)
t_test = convert(Vector{Float32}, collect(Int(round(split_ration*size(data, 1))):size(data, 1)))
t_train = convert(Vector{Float32}, collect(1:Int(round(split_ration*size(data, 1)))))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(X_train))

# Interpolation of given data
y_zoh = ConstantInterpolation(X_train, tsteps)

i = 2
state = 2
u0 = [X_train[1], mean(X_train)]

rng1 = StableRNG(i+1)
rng2 = StableRNG(i+2)

U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))
p, st = Lux.setup(rng1, U)

K = rand(rng2, Float32, 2)

function predictor!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    yt = y_zoh(t)
    e = yt .- û[1]
    du[1:end] =  û[1:end] .+ abs.(p.K) .* e
end
            
params = ComponentVector{Float32}(vector_field_model = p, K = K)
prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat = 1.0f0 )
soln_nn = Array(solve(prob_nn, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 1.0f0 ))

function prediction(p)
    _prob = remake(prob_nn, u0 = u0, p = p)
    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
    #Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = tsteps , sensealg = sensealg))
    Array(solve(_prob, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = tsteps, sensealg = sensealg))
    
end

function predloss(p)
    yh = prediction(p)
    e2 = mean(abs2, X_train .- yh[1,:])
    return e2
end
                
predloss(params)
                
losses = Float32[]
K = []

callback = function (p, l)
    push!(losses, predloss(p))
    push!(K, p.K[1:end])
                
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000, verbose = false, callback=callback) 
# Doesn't work at 5000 with AutoTsit5(Rosenbrock23(autodiff = true))- maxiters/stiffness problems reported. Set to 550, which works. AutoVern7(KenCarp4(autodiff = true)) works at 5000 iterations
# The abstol and reltol is also changed from 10e-8 to 10e-6

losses_df = DataFrame(losses = losses)
#CSV.write("sim-F1-PEM/Loss Data/Losses $i.csv", losses_df, writeheader = false)
            

full_traj = prediction(res_ms.u)

function plot_results(t, real, pred)
    plot(t, pred[1,:], label = "Training Prediction", title="PEM Model on F1 data", xlabel = "Time", ylabel = "Speed")
    plot!(t, real, label = "Training Data")
    plot!(legend=:topright)
    #savefig("sim-F1-PEM/Plots/Simulation $i.png")
end

plot_results(tsteps, X_train, full_traj)


#################################################################################################################################################
# Testing

X_train
X_test
all_data = vcat(X_train, X_test)
X_test 
t_test
tspan_test = (t_test[1], t_test[end])
tsteps_test = range(tspan_test[1], tspan_test[2], length = length(X_test))

function simulator!(du,u,p,t)
    û = U(u, p.vector_field_model, st)[1]
    du[1:end] =  û[1:end]
end

u0 = [X_test[1], mean(X_test)]
params_test = ComponentVector{Float32}(vector_field_model = p)
prob_test = ODEProblem(simulator!, u0 , tspan_test, params_test, saveat=tsteps_test)
prob = remake(prob_test, p = res_ms.u, tspan = tspan_test)
soln_nn = Array(solve(prob, AutoVern7(KenCarp4(autodiff=true)), abstol = 1f-6, reltol = 1f-6, saveat = 1.0f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))

t1 = t_train |> collect
t3 = t_test |> collect


function plot_results(train_t, test_t, train_x, test_x, train_pred, test_pred)
    plot(train_t, train_pred[1,:], label = "Training Prediction", title="Training and Test Predictions of PEM Model", xlabel = "Time", ylabel = "Population")
    plot!(test_t, test_pred[1,:], label = "Test Prediction")
    scatter!(train_t, train_x, label = "Training Data")
    scatter!(test_t, test_x, label = "Test Data")
    vline!([test_t[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/HL/PEM HL Training and Testing.png")
end

plot_results(t_train, t_test, X_train, X_test, full_traj, soln_nn)