#=

Simulation file for the Full Simulation of the NPEM Model on the Hare and Lynx data
Results were used in the master's thesis of the author - "Novel Deep Learning Strategies for Time Series Forecasting"
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-23

=#
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

# Loading the data
data_path = "case_studies/Hare and Lynx (HL)/data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame, header = true)

#Train/test Splits
split_ratio = 0.25
train = data[1:Int(round(split_ratio*size(data, 1))), :]
test = data[Int(round(split_ratio*size(data, 1))):end, :]

# Data Cleaning, Normalization and Definition
t = 0.0:1.0:581
speed = data[:,1]
train_data = convert(Vector{Float64}, train[:,2])
test_data = convert(Vector{Float64}, test[:,2])
transformer = fit(ZScoreTransform, train_data)
X_train = StatsBase.transform(transformer, train_data)
X_test = StatsBase.transform(transformer, test_data)
t_test = convert(Vector{Float64}, collect(Int(round(split_ratio*size(data, 1))):size(data, 1)))
t_train = convert(Vector{Float64}, collect(1:Int(round(split_ratio*size(data, 1)))))
tspan = (t_train[1], t_train[end])
tsteps = range(tspan[1], tspan[2], length = length(X_train))

# Interpolation of data
y_zoh = LinearInterpolation(X_train, tsteps)

# Defining the number of iterations wanted for the full simulation
iters = 2

# Defining the experimental parameters
state = 2 # Total number of states used for prediction - always one more than observed state, due to augmentation

@time begin
    for i in 1:iters

        println("Simulation $i")

        # Generating random numbers
        rng1 = StableRNG(i+1)
        rng2 = StableRNG(i+2)

        # Randomly initialize the gain parameter
        K = rand(rng2, Float64, 2)

        # Simple neural network to predict system dynamics
        U = Lux.Chain(Lux.Dense(state, 30, tanh),Lux.Dense(30, state))

        # Get the initial parameters and state variables of the model
        p, st = Lux.setup(rng1, U)

        # Predictor function for training the model
        function predictor!(du,u,p,t)
            û = U(u, p.vector_field_model, st)[1]
            yt = y_zoh(t)
            e = yt .- û[1]
            du[1:end] =  û[1:end] .+ abs.(p.K) .* e
        end

        # Constructing the ODE Problem
        u0 = [X_train[1], mean(X_train)]
        params = ComponentVector{Float64}(vector_field_model = p, K = K)
        prob_nn = ODEProblem(predictor!, u0 , tspan, params, saveat = 1.0)
        soln_nn = Array(solve(prob_nn, Tsit5(), abstol = 1e-8, reltol = 1e-8, saveat = 1.0))

        # Predictor function for training the model
        function prediction(p)
            _prob = remake(prob_nn, u0 = u0, p = p)
            sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))
            #Array(solve(_prob, AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1e-8, reltol = 1e-8, saveat = tsteps , sensealg = sensealg))
            Array(solve(_prob, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = tsteps, sensealg = sensealg))
            
        end

        # Loss function for training the model
        function predloss(p)
            yh = prediction(p)
            e2 = mean(abs2, X_train .- yh[1,:])
            return e2
        end
        
        # Defining arrays to store loss data and K-value for each iteration
        losses = Float64[]
        Ks = []

        # Callback function to store loss data and K-value for each iteration, in addition to prints the loss after every 50 iterations to facilitate monitoring
        callback = function (p, l)
            push!(losses, predloss(p))
            push!(Ks, p.K[1:end])
                        
            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")
            end
            return false
        end

        # Training the model
        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p) -> predloss(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, params)
        @time res_ms = Optimization.solve(optprob, ADAM(), maxiters = 5000, verbose = false, callback=callback) 
        # Doesn't work at 5000 with AutoTsit5(Rosenbrock23(autodiff = true))- maxiters/stiffness problems reported. Set to 550, which works. AutoVern7(KenCarp4(autodiff = true)) works at 5000 iterations
        # The abstol and reltol is also changed from 10e-8 to 10e-6

        # Saving the loss data for each iteration, in addition to the recorded K-values
        losses_df = DataFrame(losses = losses)
        Ks_mat =  mapreduce(permutedims, vcat, Ks)
        Ks_df = DataFrame(Ks_mat, :auto)
        CSV.write("full-sim-HL-NPEM/Loss Data/Losses $i.csv", losses_df, writeheader = false)
        CSV.write("full-sim-HL-NPEM/Ks/Ks $i.csv", Ks_df, writeheader = false)
                    
        # Gathering the predictions from the trained model
        full_traj = prediction(res_ms.u)

        # Plotting the training results
        function plot_results(t, real, pred)
            plot(t, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised NPEM Model", xlabel = "Time", ylabel = "Population")
            scatter!(t, real, label = "Training Data")
            plot!(legend=:topright)
            savefig("full-sim-HL-NPEM/Plots/Simulation $i.png")
        end

        plot_results(tsteps, X_train, full_traj)

    end
end

