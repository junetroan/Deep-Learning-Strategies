#ANODE-MS Model 2
using Pkg
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, StableRNGs , Plots
gr()

# Set a random seed for reproducible behaviour
rng = StableRNG(1111)

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0f0, 10.0f0)
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
println("solving ODE")

@time solution = solve(prob, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t
x̄ = mean(X, dims = 2)
noise_magnitude = 10f-3
noise_magnitude2 = 62.3f-2
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])

state = 2
groupsize = 5
predsize = 5

# Simple NN to predict lotka-volterra dynamics
U = Lux.Chain(Lux.Dense(state, 30, tanh), Lux.Dense(30, state))
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)

# Simple NN to predict initial points for use in multiple-shooting training
U0_nn = Lux.Chain(Lux.Dense(groupsize, 30, tanh), Lux.Dense(30, state-1))
p0, st0 = Lux.setup(rng, U0_nn)

# Define the hybrid model
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p.vector_field_model, st)[1] # Network prediction
    du[1:end] = û[1:end]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)

# Construct ODE Problem
augmented_u0 = vcat(Xₙ[1,:], randn(rng, Float32, 1))
params = ComponentVector{Float32}(vector_field_model = p, initial_condition_model = p0)
prob_nn = ODEProblem(nn_dynamics!, augmented_u0, tspan, params, saveat = 0.25f0)

plot(prob_nn.u0)

# Splits the full trajectory into 1) shooting segments (targets)
function group_x(X::Vector, groupsize, predictsize)
    parent = [X[i: i + max(groupsize, predictsize) - 1] for i in 1:(groupsize-1):length(X) - max(groupsize, predictsize) + 1]
    parent = reduce(hcat, parent)
    targets = parent[1:groupsize,:]
    nn_predictors = parent[1:predictsize,:]
    u0 = parent[1, :]
    return parent, targets, nn_predictors, u0
end

pas, targets, nn_predictors, u0_vec = group_x(Xₙ[1,:], groupsize, predsize)






#Checking loop
for i in 1:(groupsize-1):length(Xₙ[1,:]) - max(groupsize, predsize) + 1
    println(i)
    println(i + max(groupsize, predsize) - 1)
end

u0_nn1 = U0_nn(nn_predictors[:, 1], params.initial_condition_model, st0)[1] #1-element Vector{Float32}
u0_nn2 = U0_nn(nn_predictors[:, 2], params.initial_condition_model, st0)[1]

u0vec1 = u0_vec[1] #number
u0vec2 = u0_vec[2]

u0all = vcat(u0vec1, u0_nn1) #2-element Vector{Float32}

function prob_func(prob, i, repeat, θ)
    u0_nn = U0_nn(nn_predictors[:, i], θ.initial_condition_model, st0)[1]
    u0_all = vcat(u0_vec[i], u0_nn)
    remake(prob, u0 = u0_all, tspan = (t[1], t[groupsize]))
end

prob_func(prob_nn, 3, 1, params)

function predict(θ)
    function prob_func(prob, i, repeat)
        u0_nn = U0_nn(nn_predictors[:, i], θ.initial_condition_model, st0)[1]
        u0_all = vcat(u0_vec[i], u0_nn)
        remake(prob, u0 = u0_all, tspan = (t[1], t[groupsize]))
    end
    #sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true))
    sensealg = ReverseDiffAdjoint()
    shooting_problem = EnsembleProblem(prob = prob_nn, prob_func = prob_func) 
    Array(solve(shooting_problem, verbose = false,  AutoTsit5(Rosenbrock23(autodiff=false)), abstol = 1f-6, reltol = 1f-6, 
    p=θ, trajectories = length(u0_vec), sensealg = sensealg))#[1,:,:]
end

preds =  predict(params)

prediction_error = mean(abs2, targets .- preds[1,:,:])


function loss(θ)
    X̂ = predict(θ)
    continuity = mean(abs2, X̂[:, end, 1:end - 1] - X̂[:, 1, 2:end])
    prediction_error = mean(abs2, targets .- X̂[1,:,:])
    prediction_error + continuity*10f0
end

U0_nn(nn_predictors[:,1], params.initial_condition_model, st0)[1]

function predict_final(θ)
    predicted_u0_nn = U0_nn(nn_predictors[:,1], θ.initial_condition_model, st0)[1]
    u0_all = vcat(u0_vec[1], predicted_u0_nn)
    prob_nn_updated = remake(prob_nn, p=θ, u0=u0_all) # no longer updates u0 nn
    X̂ = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff=true)), 
    abstol = 1f-6, reltol = 1f-6, 
    saveat = 0.25f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
    X̂
end

function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, Xₙ[1,:] .- X̂[1, :])
    prediction_error
end

losses = Float32[]
callback = function (θ, l)
    push!(losses, final_loss(θ))

    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, params)
@time res_ms = Optimization.solve(optprob, ADAM(), callback=callback, maxiters = 5000)

final_traj = predict_final(res_ms.u)[1,:]
plot(final_traj)
scatter!(Xₙ[1,:])

full_traj_loss = final_loss(res_ms.u)
println("Full Trajectory Loss: ",full_traj_loss)

optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), callback=callback, maxiters = 1000, allow_f_increases = true)

print(losses[end])

full_traj2 = predict_final(res_final.u)
actual_loss = Xₙ[1,:] - full_traj2[1,:]
total_loss = abs(sum(actual_loss))

plot(full_traj2[1, :])
scatter!(Xₙ[1, :])

#############################################################################################
# Testing 
p_ = Float32[1.3, 0.9, 0.8, 1.8]
prob_new = ODEProblem(lotka!, u0, (0.0f0, 40.0f0), p_)
@time solution_new = solve(prob_new, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0)


predicted_u0_nn = U0_nn(nn_predictors[:, 1], res_final.u.initial_condition_model, st0)[1]
u0_all = vcat(u0_vec[1], predicted_u0_nn)
prob_nn_updated = remake(prob_nn, p = res_final.u, u0 = u0_all, tspan = (0.0f0, 40.0f0))
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)),  abstol = 1f-6, reltol = 1f-6,
saveat = 0.25f0, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))))
updated_obsgrid = 0.0:0.25:40.0
t1 = t
t3 = updated_obsgrid[41:123] |> collect

gr()
function plot_results(t, real, pred, pred_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Training and Test Predictions of ANODE-MS Model", xlabel = "Time", ylabel = "Population")
    plot!(t3, pred_new[1,41:123], label = "Test Prediction")
    scatter!(t1, real[1,:], label = "Training Data")
    scatter!(t3, solution_new[1,41:123], label = "Test Data")
    vline!([t1[41]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/ANODE-MS LV Training and Testing.png")
end

plot_results(t, Xₙ, full_traj2, prediction_new)

actual_loss = solution_new[1,:] - prediction_new[1,:]
total_loss = abs(sum(actual_loss))
total_loss2 = abs2(sum(actual_loss))
MSE = total_loss2 / length(solution_new[1,:])

t_all = updated_obsgrid |> collect

function plot_residuals(t,residuals)
    plot(t, residuals, label = "Residuals", xlabel= "Time", ylabel="Residuals", title="Residuals over time of ANODE-MS Model")
    #savefig("Results/ms-residuals-time.png")
end

plot_residuals(t_all,actual_loss)

function plot_loss(calculated_loss)
    plot(calculated_loss, xlabel="Iteration", ylabel="Loss", legend=false, yscale=:log10, title="Loss evolution of ANODE-MS Model")
    #plot(calculated_loss, xlabel="Iteration", ylabel="Loss", legend=false, ylims=(0,5))
    #savefig("Results/ms-loss-per-iteration-logy.png")
end

plot_loss(losses)

function plot_histogram(residuals)
    histogram(residuals, title="Histogram of residuals of ANODE-MS Model", label="Residuals", xlabel="Residuals", ylabel="Frequency")
    #savefig("Results/ms-histogram-residuals.png")
end

plot_histogram(actual_loss)

function plot_scatter(predicted, real)
    plot([minimum(predicted[1,:]), maximum(predicted[1,:])], [minimum(predicted[1,:]), maximum(predicted[1,:])], label="y=x", color=:red, title="Parity plot for the ANODE-MS model")
    scatter!(real[1,:], predicted[1,:], label = "Residuals", xlabel= "Actual values", ylabel="Predicted values", color=RGB(165/225,113/225,220/225))
    savefig("Results/ms-parity.png")
end

plot_scatter(prediction_new, solution_new)
=#