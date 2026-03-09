import matplotlib.pyplot as plt

# Data (ascending sampling rate)
sampling_rates = [125, 250, 500, 1000, 2000]
cer = [73.8275375366211, 36.740867614746094, 29.933002471923828, 25.58893394470215, 22.64067268371582]

plt.figure(figsize=(6,4))

plt.plot(sampling_rates, cer, marker='o', linewidth=2)

plt.xlabel("Sampling Rate (Hz)")
plt.ylabel("Character Error Rate (CER)")
plt.title("Effect of EMG Sampling Rate on Typing Decoding Performance")

plt.xticks(sampling_rates)
plt.grid(True)

# Save figure
plt.savefig("sampling_rate_vs_cer.png", dpi=300, bbox_inches="tight")

plt.show()