using CSV
using DataFrames
using Glob
using Plots

# Directory containing the CSV files
folder_path = "/Users/junetroan/Desktop/Results/Ks"

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
    plot!([i, i], [initial_first_values[i], final_first_values[i]], line = (:blue, 5), label = "")
end
title!("Change in K Over Simulations")
xlabel!("Simulation")
ylabel!("K Value")
savefig("K_value_changes.png")
