import matplotlib.pyplot as plt
import numpy as np

# Data
matrix_sizes_format = [
    '1024x1024, 1024x1024, 1024x1024', '2048x1024, 1024x1024, 2048x1024', 
    '1024x2048, 2048x1024, 1024x1024', '2048x2048, 2048x1024, 2048x1024',
    '2048x2048, 2048x2048, 2048x2048', '3072x2048, 2048x2048, 3072x2048', 
    '3072x3072, 3072x2048, 3072x2048', '3072x3072, 3072x3072, 3072x3072',
    '4096x3072, 3072x3072, 4096x3072', '4096x4096, 4096x3072, 4096x3072'
]

gpu_kernel_times = [
    0.001871, 0.004073, 0.00349, 0.006619, 0.012946, 0.019279, 
    0.030173, 0.047915, 0.063727, 0.087135
]

copy_to_gpu_times = [
    0.00109, 0.002145, 0.002256, 0.004007, 0.004555, 0.007034, 
    0.008716, 0.011867, 0.01312, 0.01758
]

copy_to_cpu_times = [
    0.001498, 0.002825, 0.001393, 0.002684, 0.004679, 0.00747, 
    0.006739, 0.010118, 0.013229, 0.013455
]

cpu_times = [
    4.503231, 8.97181, 10.721591, 17.108295, 43.105496, 64.326178, 
    114.00401, 148.475776, 194.402994, 491.909284
]

# Plot with only GPU times (excluding Total GPU Time) with logarithmic scale
plt.figure(figsize=(14, 7))
plt.semilogy(matrix_sizes_format, gpu_kernel_times, marker='o', label='GPU Kernel Time')
plt.semilogy(matrix_sizes_format, copy_to_gpu_times, marker='s', label='Copy to GPU Time')
plt.semilogy(matrix_sizes_format, copy_to_cpu_times, marker='^', label='Copy to CPU Time')
plt.title('GPU Times for Various Matrix Sizes (Logarithmic Scale)')
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds, log scale)')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('float-gpu_times_log_scale.png')

# Plot with both GPU and CPU times (excluding Total GPU Time) with logarithmic scale
plt.figure(figsize=(14, 7))
plt.semilogy(matrix_sizes_format, gpu_kernel_times, marker='o', label='GPU Kernel Time')
plt.semilogy(matrix_sizes_format, copy_to_gpu_times, marker='s', label='Copy to GPU Time')
plt.semilogy(matrix_sizes_format, copy_to_cpu_times, marker='^', label='Copy to CPU Time')
plt.semilogy(matrix_sizes_format, cpu_times, marker='x', label='CPU Computation Time', color='red')
plt.title('GPU and CPU Times for Various Matrix Sizes (Logarithmic Scale)')
plt.xlabel('Matrix Size')
plt.ylabel('Time (seconds, log scale)')
plt.xticks(rotation=90)
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('float-gpu_and_cpu_times_log_scale.png')