#=

Simulation file for the Full Simulation of the MNODE-MS Model on the Hare and Lynx data
Results were used in the master's thesis of the author - "Novel Deep Learning Strategies for Time Series Forecasting"
Author: June Mari Berge Trøan (@junetroan)
Last updated: 2024-05-23

=#

# Loading libraries
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

# Loading the data
data_path = "case_studies/Hare and Lynx (HL)/data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame)

#Train/test Splits
split_ratio = 0.25
train = data[1:Int(round(split_ratio*size(data, 1))), :]
test = data[Int(round(split_ratio*size(data, 1))):end, :]

# Data Cleaning, Normalization and Definition
hare_data = data[:, 2]
lynx_data = data[:, 3]
transformer = fit(ZScoreTransform, hare_data)
X_train = Float64.(StatsBase.transform(transformer, train[:, 2]))
datasize = size(X_train, 1)
X_test = Float64.(StatsBase.transform(transformer, test[:, 2]))
t = Float64.(collect(1:size(data, 1)))
t_train = Float64.(collect(1:Int(round(split_ratio*size(data, 1)))))
t_test = Float64.(collect(Int(round(split_ratio*size(data, 1))):size(data, 1)))
tspan = (Float64(minimum(t_train)), Float64(maximum(t_train)))
tsteps = range(tspan[1], tspan[2], length = datasize)
step = 1.0f0
x = X_train

# Defining the number of iterations wanted for the full simulation
iters = 2

# Defining the experimental parameters
group_size = 5 # Number of points in each trajectory
state = 2 # Total number of states used for prediction - always one more than observed state, due to augmentation
continuity_term = 10.0 # Define the continuity factor that penalises the difference between the last state in the previous prediction and the current initial condition


@time begin
    for i in 1:iters

        println("Simulation $i")

        # Generating random numbers
        rng1 = StableRNG(i)

         # Simple neural network to predict system dynamics
        U = Lux.Chain(Lux.Dense(state, 30, tanh),
        Lux.Dense(30, state))

        # Get the initial parameters and state variables of the model
        p, st = Lux.setup(rng1, U)

        # Constructing the ODE Problem
        params = ComponentVector{Float64}(vector_field_model = p)
        neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, ComponentArray(p))

        # Grouping the data into trajectories for multiple shooting
        function group_ranges(datasize::Integer, groupsize::Integer)
            2 <= groupsize <= datasize || throw(DomainError(groupsize,
            "datasize must be positive and groupsize must to be within [2, datasize]"))
            return [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
        end

        # Gathering the trajectories and intial conditions
        ranges = group_ranges(datasize, group_size)
        u0 = Float64(X_train[first(1:5)])
        u0_init = [[X_train[first(rg)]; fill(mean(X_train[rg]), state - 1)] for rg in ranges] 
        u0_init = mapreduce(permutedims, vcat, u0_init)
        
        # Modified multiple shooting method
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
                solver, saveat = tsteps[rg], sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))) 
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

        # Define the loss function and continuity loss function
        loss_function(data, pred) = sum(abs2, data - pred)
        continuity_loss(uᵢ₊₁, uᵢ) = sum(abs2, uᵢ₊₁ - uᵢ)

        # Defining the prediction function utilising single shooting method
        predict_single_shooting(p) = Array(first(neuralode(u0_init[1,:],p,st)))
        tester_pred_ss = predict_single_shooting(params.vector_field_model)

        # Loss function for the single shooting method
        function loss_single_shooting(p)
            pred = predict_single_shooting(p)
            l = loss_function(x, pred[1,:])
            return l, pred
        end

        # Gathering the prediction of the single shooting method
        ls, ps = loss_single_shooting(params.vector_field_model)

        # Defining the parameters for the prediction function utilising the multiple shooting method
        params = ComponentVector{Float64}(θ = p, u0_init = u0_init)
        ls, ps = multiple_shoot_mod(params, x, tsteps, prob_node, loss_function, continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size; continuity_term)

        # Running the prediction function utilising the mulitple shooting method
        multiple_shoot_mod(params, x, tsteps, prob_node, loss_function,
            continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size;
            continuity_term)

        # Defining the loss function for the multiple shooting method
        function loss_multiple_shoot(p)
            return multiple_shoot_mod(p, x, tsteps, prob_node, loss_function,
                continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size;
                continuity_term)[1]
        end
        
        # Defining an array to store loss data for each iteration
        losses = Float64[]

        # Callback that prints the loss after every 50 iterations to facilitate monitoring
        callback = function (θ, l)
            push!(losses, loss_multiple_shoot(θ))
            if length(losses) % 50 == 0
                println("Current loss after $(length(losses)) iterations: $(losses[end])")

            end
            return false
        end

        # Training the model
        adtype = Optimization.AutoZygote()
        optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shoot(x), adtype)
        optprob = Optimization.OptimizationProblem(optf, params)
        @time res_ms = Optimization.solve(optprob, ADAM(),  maxiters = 5000, callback = callback) #callback  = callback,

        # Saving the loss data
        losses_df = DataFrame(loss = losses)
        CSV.write("full-sim-HL-MNODE-MS/Losses $i.csv", losses_df, writeheader = false)

        # Gathering the predictions from the trained model and calculating the loss
        loss_ms, _ = loss_single_shooting(res_ms.u.θ)
        preds = predict_single_shooting(res_ms.u.θ)

        # Plotting the training results
        function plot_results(real, pred)
            plot(t_train, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised MNODE-MS Model", xlabel="Time", ylabel="Population")
            scatter!(t_train, real, label = "Training Data")
            plot!(legend=:topright)
            savefig("full-sim-HL-MNODE-MS/Plots/Simulation $i.png")
        end

        plot_results(x, preds)

        if i == iters
            println("Simulation finished")
            break 
        end

    end
end

