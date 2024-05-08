using CSV
using DataFrames
using Glob
using Plots

PlotlyKaleido.start()
plotly()

# Directory containing the CSV files
folder_path = "/Users/junetroan/Desktop/NPEM LV/Ks"

# List all CSV files in the directory
csv_files = glob("*.csv", folder_path)

# Initialize arrays to store the initial and final K values for each experiment
initial_first_values = Float64[]
final_first_values = Float64[]

# Process each CSV file
for file in csv_files
    # Read the CSV file into a DataFrame
    df = CSV.read(file, DataFrame)

    # Convert the first and last rows to strings (to ensure compatibility with split)
    initial_row = string(df[1, 1])   # Convert first row to string
    final_row = string(df[end, 1])   # Convert last row to string

    # Split the values and convert to float
    initial_vals = split(initial_row, ',')
    final_vals = split(final_row, ',')

    # Append the first value (K) from the initial and final rows to the arrays
    push!(initial_first_values, parse(Float64, initial_vals[1]))
    push!(final_first_values, parse(Float64, final_vals[1]))
end

# Create the plot with bars for each experiment
p = plot(bar_width = 0.7, legend = false)
for i in 1:length(initial_first_values)
    # Plot each bar from the initial to final value
    plot!([i, i], [initial_first_values[i], final_first_values[i]], line = (:orange, 5), label = "")
end
title!("Change in K Over Simulations")
xlabel!("Simulation")
ylabel!("K Value")
#savefig("K_value_changes.png")



# Examination of extreme values/changes in Ks

# Small K_value_changes
small_changes = filter(x -> 0 < x < 0.003, final_first_values .- initial_first_values)
iterations_small_changes = findall(x -> 0 < x < 0.004, final_first_values .- initial_first_values)

initial_first_values[18]
final_first_values[18]

loss_18 = CSV.read("/Users/junetroan/Desktop/NPEM LV/Loss Data/Losses 18.csv", DataFrame)

p_18 = plot(1:nrow(loss_18), loss_18[!,1], title="Loss Evolution for Simulation 18", xlabel="Iteration", ylabel="Loss", legend=false)
savefig(p_18, "Results/LV/NPEM/Loss Evolution for Simulation 18.png")

start_loss_18 = loss_18[1,1]
end_loss_18 = loss_18[end,1]
change_loss_18 = abs(end_loss_18 - start_loss_18)

initial_first_values[170]
final_first_values[170]

loss_170 = CSV.read("/Users/junetroan/Desktop/NPEM LV/Loss Data/Losses 170.csv", DataFrame)

p_170 = plot(1:nrow(loss_170), loss_170[!,1], title="Loss Evolution for Simulation 170", xlabel="Iteration", ylabel="Loss", legend=false)
savefig(p_170, "Results/LV/NPEM/Loss Evolution for Simulation 170.png")

start_loss_170 = loss_170[1,1]
end_loss_170 = loss_170[end,1]
change_loss_170 = abs(end_loss_170 - start_loss_170)

# Large K_value_changes
large_changes = filter(x -> x > 0.5, final_first_values .- initial_first_values)
iterations_large_changes = findall(x -> x > 0.5, final_first_values .- initial_first_values)

initial_first_values[298]
final_first_values[298]

loss_298 = CSV.read("/Users/junetroan/Desktop/NPEM LV/Loss Data/Losses 298.csv", DataFrame)

p_298 = plot(1:nrow(loss_298), loss_298[!,1], title="Loss Evolution for Simulation 298", xlabel="Iteration", ylabel="Loss", legend=false)
savefig(p_298, "Results/LV/NPEM/Loss Evolution for Simulation 298.png")

start_loss_298 = loss_298[1,1]
end_loss_298 = loss_298[end,1]
change_loss_298 = abs(end_loss_298 - start_loss_298)

initial_first_values[438]
final_first_values[438]

loss_438 = CSV.read("/Users/junetroan/Desktop/NPEM LV/Loss Data/Losses 438.csv", DataFrame)

p_438 = plot(1:nrow(loss_438), loss_438[!,1], title="Loss Evolution for Simulation 438", xlabel="Iteration", ylabel="Loss", legend=false)
savefig(p_438, "Results/LV/NPEM/Loss Evolution for Simulation 438.png")

start_loss_438 = loss_438[1,1]
end_loss_438 = loss_438[end,1]
change_loss_438 = abs(end_loss_438 - start_loss_438)


