# Test - Lynx Hare PEM 

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

data_path = "Multiple Shooting (MS)/ANODE-MS/Case Studies/Financial System/financial_time_series.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
split_ration = 0.75
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
t = data[:,1]
ir = data[:,2]

transformer = fit(ZScoreTransform, ir)
X_train = StatsBase.transform(transformer, train[:, 2])
X_test = StatsBase.transform(transformer, test[:, 2])
t_test = collect(Int(round(split_ration*size(data, 1))):size(data, 1))
t_train = collect(1:Int(round(split_ration*size(data, 1))))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(X_train))

# Interpolation of given data
y_zoh = ConstantInterpolation(X_train, tsteps)

# Definition of neural network
iters = 2
state = 2
u0 = [X_train[1],0]

@time begin
    for i in 1:iters

        println("Simulation $i")

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
        soln_nn = Array(solve(prob_nn, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = 1.0f0 ))

        function prediction(p)
            _prob = remake(prob_nn, u0 = u0, p = p)
            sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
            Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = tsteps , sensealg = sensealg))
        end
        
        prediction(params)
        
        function predloss(p)
            yh = prediction(p)
            e2 = mean(abs2, X_train .- yh[1,:])
            return e2
        end
        
        predloss(params)
        
        losses = Float32[]
        callback = function (p, l)
            push!(losses, predloss(p))
        
            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
            end
            return false
        end

        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, params)
        res_ms = Optimization.solve(optprob, ADAM(), maxiters = 10000, verbose = false, callback=callback)
        
        losses_df = DataFrame(losses = losses)
        CSV.write("Multiple Shooting (MS)/ANODE-MS/Simulations/Results/sim-FS-PEM/Loss Data/Losses $i.csv", losses_df, writeheader = false)
        
        full_traj = prediction(res_ms.u)
        full_traj_loss = predloss(res_ms.u)
        push!(fulltraj_losses, full_traj_loss)
        
        function plot_results(t, real, pred)
            plot(t, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised PEM Model", xlabel = "Time", ylabel = "Population")
            scatter!(t, real, label = "Training Data")
            plot!(legend=:topright)
            savefig("Multiple Shooting (MS)/ANODE-MS/Simulations/Results/sim-FS-PEM/Plots/Simulation $i.png")
        end
        
        plot_results(t, X_train, full_traj)

    end
end