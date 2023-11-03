import matplotlib.pyplot as plt

# Data from the CUDA outputs
lengths = [
    1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 
    262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216
]

time_h2d = [
    0.231213, 0.268904, 0.282739, 0.273542, 0.215975, 0.323556, 
    0.213930, 0.321532, 0.532277, 1.218241, 2.496176, 5.652745, 
    9.763069, 22.726673, 43.988931
]

time_kernel = [
    0.095329, 0.100899, 0.083857, 0.082284, 0.096080, 0.087954, 
    0.111980, 0.150271, 0.243195, 0.262161, 0.296234, 0.383959, 
    0.580727, 0.952955, 1.871444
]

time_d2h = [
    0.147055, 0.056536, 0.246432, 0.147957, 0.097512, 0.214972, 
    0.230702, 0.499074, 0.658994, 1.384442, 2.499159, 5.191940, 
    9.014196, 19.543090, 42.464435
]

# Re-create the line plot with the updated data
plt.figure(figsize=(12, 8))

# Plotting each operation with updated data
plt.plot(lengths, time_h2d, marker='o', label='Host to Device')
plt.plot(lengths, time_kernel, marker='s', label='Kernel Execution')
plt.plot(lengths, time_d2h, marker='^', label='Device to Host')

# Adding labels and title
plt.xlabel('Vector Length')
plt.ylabel('Time (ms)')
plt.title('CUDA Operations Time as a Function of Vector Length with Updated Data')
plt.xscale('log')  # Log scale for x-axis to see the increase in vector length more clearly
plt.yscale('log')  # Log scale for y-axis due to wide range of values
plt.xticks(lengths, [str(length) for length in lengths])
plt.grid(True, which="both", ls="--")
plt.legend()

# Show the plot with a tight layout
plt.tight_layout()
plt.savefig('exercise1.png')