# Test againt Julias multiple shooting
using Pkg
using DifferentialEquations
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Distributions
using ComponentArrays, Lux, DiffEqFlux, Optimization, OptimizationPolyalgorithms, DifferentialEquations, Plots
using DiffEqFlux: group_ranges
using OptimizationOptimisers
using StableRNGs

rng = StableRNG(1111)
# Define initial conditions and time steps
tspan = (0.0f0, 10.0f0) # Original (0.0f0, 10.0f0)
tsteps = 0.25f0
u0 = 5.0f0 * rand(rng, Float32, 2)
p_ = Float32[1.3, 0.9, 0.8, 1.8]

datasize = 41
tsteps = range(tspan[1], tspan[2], length = datasize)

# Get the data
function trueODEfunc(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan, p_)
ode_data = Array(solve(prob_trueode, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = tsteps))
x̄ = mean(ode_data, dims = 2)
noise_magnitude = 10f-3
Xₙ = ode_data .+ (noise_magnitude * x̄) .* randn(rng, eltype(ode_data), size(ode_data))

# Define the Neural Network
nn = Lux.Chain(Lux.Dense(2, 30, tanh), Lux.Dense(30, 2))
p_init, st = Lux.setup(rng, nn)

neuralode = NeuralODE(nn, tspan, AutoTsit5(Rosenbrock23(autodiff = false)), saveat = tsteps, sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)))
prob_node = ODEProblem((u,p,t) -> nn(u,p,st)[1], u0, tspan, ComponentArray(p_init))

# Define parameters for Multiple Shooting
group_size = 5
continuity_term = 10.0f0

function loss_function(data, pred)
	return sum(abs2, data - pred)
end

function loss_multiple_shooting(p)
    return multiple_shoot(p, Xₙ, tsteps, prob_node, loss_function, AutoTsit5(Rosenbrock23(autodiff=false)),
                          group_size; continuity_term)
end

adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x,p) -> loss_multiple_shooting(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentArray(p_init))
@time res_ms = Optimization.solve(optprob, ADAM(), maxiters=5000)



function predict_single_shooting(p)
    return Array(neuralode(u0, p, st)[1])
end

# Define loss function
function loss_function(data, pred)
	return sum(abs2, data - pred)
end

## Evaluate Single Shooting
function loss_single_shooting(p)
    pred = predict_single_shooting(p)
    l = loss_function(Xₙ, pred)
    return l, pred
end

loss_ms, pr = loss_single_shooting(res_ms.minimizer)
println("Multiple shooting loss: $(loss_ms)")

#TODO: Find way to update u0
function predict_final(θ)
    return Array(neuralode(u0, θ, st)[1])
end

full_traj = predict_final(res_ms.u)
plot(full_traj[1, :])
scatter!(Xₙ[1, :])

function final_loss(θ)
    X̂ = predict_final(θ)
    prediction_error = mean(abs2, Xₙ[1,:] .- X̂[1, :])
    prediction_error
end

optf_final = Optimization.OptimizationFunction((x,p) -> final_loss(x), adtype)
optprob_final = Optimization.OptimizationProblem(optf_final, res_ms.u)
res_final = Optimization.solve(optprob_final, BFGS(initial_stepnorm = 0.01), maxiters = 1000, allow_f_increases = true)

full_traj2 = predict_final(res_final.u)
plot(full_traj2[1, :])


actual_loss = Xₙ[1,:] - full_traj2[1,:]
total_loss = abs2(sum(actual_loss))


###############################################################################################
#Testing
tspan2 = (0.0f0, 40.0f0)
prob_new = ODEProblem(trueODEfunc, u0, (0.0f0, 40.0f0), p_)
solution_new = Array(solve(prob_new, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = 0.25f0))

prob_nn_updated = remake(prob_node, p = res_final.u, u0 = u0, tspan = (0.0f0, 40.0f0))
#prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4()), abstol = 1e-8, reltol = 1e-8, saveat = tsteps))
prediction_new = Array(solve(prob_nn_updated, AutoVern7(KenCarp4(autodiff = true)),  
abstol = 1f-6, reltol = 1f-6,saveat = 0.25f0))

plot(prediction_new[1, :])
scatter!(solution_new[1, :])


###############################################################################################

gr()
updated_obsgrid = 0.0:0.25:40.0
t1 = tsteps[1:41] |> collect
t3 = updated_obsgrid[41:123] |> collect

#plot(t3, prediction_new[1,1:123], label = "Test Prediction")

function plot_results(t1, t3, real, pred, pred_new)
    plot(t1, pred[1,:], label = "Training Prediction", title="Training and Test Predictions of NODE-MS Model", xlabel="Time", ylabel="Population")
    plot!(t3, pred_new[1,41:123], label = "Test Prediction")
    scatter!(t1, real[1,:], label = "Training Data")
    scatter!(t3, prediction_new[1,41:123], label = "Test Data")
    vline!([t1[41]], label = "Training/Test Split")
    plot!(legend=:topright)
    savefig("Results/ms-julia-results-updated.png")
end

plot_results(t1,t3, Xₙ, full_traj2, prediction_new)

actual_loss = solution_new[1,:] - prediction_new[1,:]
total_loss = abs(sum(actual_loss))
total_loss2 = abs2(sum(actual_loss))    
MSE = total_loss2 / length(solution_new[1,:])


function plot_scatter(predicted, real)
    plot([minimum(predicted[1,:]), maximum(predicted[1,:])], [minimum(predicted[1,:]), maximum(predicted[1,:])], label="y=x", color=:red, title="Parity plot for the NODE-MS model")
    scatter!(real[1,:], predicted[1,:], label = "Residuals", xlabel= "Actual values", ylabel="Predicted values", color=RGB(165/225,113/225,220/225))
    savefig("Results/ms-julia-parity.png")
end

plot_scatter(prediction_new, solution_new)


#plot(prediction_new[1, :])
#=
plot(full_traj2[1, :])
scatter!(Xₙ[1, :])

actual_loss = Xₙ[1,:] - full_traj2[1,:]
total_loss = abs(sum(actual_loss))
N = datasize
MSE = (1/N)*sum((Xₙ[1,:] .- pred[1,:]).^2)

function plot_results(t, real, pred)
    plot(t, pred[1,:], label = "Prediction")
    scatter!(t, real[1,:], label = "Data")
    #savefig("Results/ms-julia-results-updated.png")
    #savefig("Results/ms-julia-results.png")
end

plot_results(tsteps, Xₙ, full_traj2)

function plot_scatter(predicted, real)
    scatter(real[1,:], predicted[1,:], label = "Residuals", xlabel= "Actual values", ylabel="Predicted values")
    plot!([minimum(predicted[1,:]), maximum(predicted[1,:])], [minimum(predicted[1,:]), maximum(predicted[1,:])], label="y=x")
    #savefig("Results/ms-julia-residuals.png")
end

plot_scatter(full_traj, Xₙ)

function plot_histogram(residuals)
    histogram(residuals)
    #savefig("Results/ms-julia-histogram-residuals.png")
end

plot_histogram(actual_loss)

function plot_loss(calculated_loss)
    #plot(calculated_loss, xlabel="Iteration", ylabel="Loss", legend=false, yscale=:log10)
    plot(calculated_loss, xlabel="Iteration", ylabel="Loss", legend=false, ylims=(0,5))
    #savefig("Results/ms-loss-per-iteration-logy.png")
end

plot_loss(actual_loss)
=#