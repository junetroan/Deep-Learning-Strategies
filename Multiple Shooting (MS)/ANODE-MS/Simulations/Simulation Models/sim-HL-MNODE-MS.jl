
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
data_path = "Multiple Shooting (MS)/ANODE-MS/Data/lynx_hare_data.csv"
data = CSV.read(data_path, DataFrame)

#Train/test Splits
split_ration = 0.25
train = data[1:Int(round(split_ration*size(data, 1))), :]
test = data[Int(round(split_ration*size(data, 1))):end, :]

# Data Cleaning and Normalization
hare_data = data[:, 2]
lynx_data = data[:, 3]

transformer = fit(ZScoreTransform, hare_data)
X_train = Float64.(StatsBase.transform(transformer, train[:, 2]))
datasize = size(X_train, 1)

plot(X_train)

X_test = Float64.(StatsBase.transform(transformer, test[:, 2]))
t = Float64.(collect(1:size(data, 1)))
t_train = Float64.(collect(1:Int(round(split_ration*size(data, 1)))))
t_test = Float64.(collect(Int(round(split_ration*size(data, 1))):size(data, 1)))
tspan = (Float64(minimum(t_train)), Float64(maximum(t_train)))
tsteps = range(tspan[1], tspan[2], length = datasize)


# Define the experimental parameter
groupsize = 5
predsize = 5
state = 2

fulltraj_losses = Float64[]

# NUMBER OF ITERATIONS OF THE SIMULATION
iters = 2

i = 1
# Random numbers
rng_1 = StableRNG(i)
rng_2 = StableRNG(i + 2)

# NEURAL NETWORK
U = Lux.Chain(Lux.Dense(state, 30, tanh),
Lux.Dense(30, state))

tsteps = 1.0f0
p, st = Lux.setup(rng_1, U)
params = ComponentVector{Float64}(vector_field_model = p)
u0 = vcat(X_train[1], randn(rng_2, state - 1))
neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, ComponentArray(p))

tsteps = range(tspan[1], tspan[2], length = length(X_train))

##############################################################################################################################################################
# Testing to reveal variable types and variable content

datasize
groupsize
group_size = groupsize
X_train
prob_node
x = X_train

####################################################################################################################################################################

# Simulation 

####################################################################################################################################################################


iters = 2

@time begin
    for i in 1:iters

        println("Simulation $i")

        # Random numbers
        rng_1 = StableRNG(i)

        # NEURAL NETWORK

        U = Lux.Chain(Lux.Dense(state, 30, tanh),
        Lux.Dense(30, state))

        p, st = Lux.setup(rng_1, U)
        params = ComponentVector{Float64}(vector_field_model = p)
        neuralode = NeuralODE(U, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
        prob_node = ODEProblem((u,p,t) -> U(u, p, st)[1][1:end], u0, tspan, ComponentArray(p))

                function group_ranges(datasize::Integer, groupsize::Integer)
                    2 <= groupsize <= datasize || throw(DomainError(groupsize,
                    "datasize must be positive and groupsize must to be within [2, datasize]"))
                    return [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
                end

                ranges = group_ranges(datasize, group_size)

                u0 = Float64(X_train[first(1:5)])
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
                params = ComponentVector{Float64}(θ = p, u0_init = u0_init)
                ls, ps = multiple_shoot_mod(params, x, tsteps, prob_node, loss_function, continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size; continuity_term)

                # Modified multiple_shoot method 
                multiple_shoot_mod(params, x, tsteps, prob_node, loss_function,
                    continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size;
                    continuity_term)

                function loss_multiple_shoot(p)
                    return multiple_shoot_mod(p, x, tsteps, prob_node, loss_function,
                        continuity_loss, AutoTsit5(Rosenbrock23(autodiff = false)), group_size;
                        continuity_term)[1]
                end
           
                losses = Float64[]

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
                @time res_ms = Optimization.solve(optprob, ADAM(),  maxiters = 5000, callback = callback) #callback  = callback,

                losses_df = DataFrame(loss = losses)
                CSV.write("Loss Data/HL Losses $i.csv", losses_df, writeheader = false)

                loss_ms, _ = loss_single_shooting(res_ms.u.θ)
                preds = predict_single_shooting(res_ms.u.θ)

                function plot_results(real, pred)
                    plot(t_train, pred[1,:], label = "Training Prediction", title="Iteration $i of Randomised MNODE-MS Model", xlabel="Time", ylabel="Population")
                    scatter!(t_train, real, label = "Training Data")
                    plot!(legend=:topright)
                    savefig("Plots/HL Simulation $i.png")
                end

                plot_results(x, preds)

                if i == iters
                    println("Simulation finished")
                    break 
                end

            end
        end



##################################################################################################################################################################

# Single round

##################################################################################################################################################################

function group_ranges(datasize::Integer, groupsize::Integer)
    2 <= groupsize <= datasize || throw(DomainError(groupsize,
    "datasize must be positive and groupsize must to be within [2, datasize]"))
    return [i:min(datasize, i + groupsize - 1) for i in 1:(groupsize - 1):(datasize - 1)]
end

ranges = group_ranges(datasize, groupsize)
u0 = Float64(X_train[first(1:5)])
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
        solver, saveat = tsteps[rg], sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))) 
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
params = ComponentVector{Float64}(θ = p, u0_init = u0_init)
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

losses = Float64[]

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
#CSV.write("Multiple Shooting (MS)/ANODE-MS/Simulations/Results/sim-LV-MNODE-MS/Loss Data/Losses $i.csv", losses_df, writeheader = false)

loss_ms, _ = loss_single_shooting(res_ms.u.θ)
preds = predict_single_shooting(res_ms.u.θ)

plot(preds[1, :])
scatter!(X_train)

gr()
function plot_results(real, pred, t)
    plot(t, pred[1,:], label = "Training Prediction", title="Trained MNODE-MS Model predicting Hare data", xlabel="Time", ylabel="Population")
    plot!(t, real, label = "Training Data")
    plot!(legend=:topleft)
    savefig("Results/HL/Training MNODE-MS Model on Hare and Lynx data.png")
end

plot_results(X_train, preds, t_train)

#####################################################################################################################################################################
# Testing 

X_test
t_test
tspan_test = (Float64(minimum(t_test)), Float64(maximum(t_test)))
tsteps_test = range(tspan_test[1], tspan_test[2], length = length(X_test))

u0 = vcat(res_ms.u.u0_init[1,:])
prob_nn_updated = remake(prob_node, u0 = u0, tspan = tspan_test, p = res_ms.u.θ)
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)), abstol = 1e-6, reltol = 1e-6, saveat = 1.0f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
t1  = t_train |> collect
t3 = t_test |> collect

test_loss = X_test - prediction_new[1,:]
total_test_loss = sum(abs, test_loss)

gr()

function plot_results(real_train, real_test, train_pred, test_pred)
    plot(t1, train_pred[1,:], label = "Training Prediction", title="Training and Test Predictions of MNODE-MS Model", xlabel = "Time", ylabel = "Population")
    plot!(t3, test_pred[1,:], label = "Test Prediction")
    scatter!(t1, real_train, label = "Training Data")
    scatter!(t3, real_test, label = "Test Data")
    vline!([t3[1]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/HL/Training and testing of MNODE-MS Model on Hare and Lynx data.png")
end

plot_results(X_train, X_test, preds, prediction_new)
