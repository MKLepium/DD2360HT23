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

gpu_kernel_times = [0.006789, 0.013862, 0.014000, 0.029465, 0.057144, 0.083154, 0.122675, 0.178354, 0.234149, 0.300778]
copy_to_gpu_times = [0.002521, 0.003523, 0.004555, 0.007985, 0.009775, 0.012426, 0.018417, 0.021488, 0.027468, 0.035143]
copy_to_cpu_times = [0.003048, 0.005791, 0.002981, 0.004892, 0.009823, 0.013589, 0.014509, 0.020851, 0.031874, 0.028040]
cpu_times = [4.489160, 9.235074, 14.409864, 21.813474, 193.810544, 285.614809, 429.450031, 650.361157, 895.589857, 1209.905773]

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
plt.savefig('gpu_times_log_scale.png')

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
plt.savefig('gpu_and_cpu_times_log_scale.png')