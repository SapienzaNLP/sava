import matplotlib.pyplot as plt

# HELLASWAG ITALIAN
mistral_ita_1e4_hellaswag = [54.63, 58.76, 59.86, 59.58, 60.96, 61.98, 62.26, 62.9, 64.41, 64.2]
mistral_en_ita_1e4_hellaswag = [56.38, 59.08, 59.84, 60.37, 61.8, 62.69, 63.91, 64.06, 65.87, 65.81]
mistral_en_ita_1e5_hellaswag = [64.1, 65.01, 65.98, 66.25, 66.07, 66.56, 66.56, 66.54, 67.13, 67.45]
mistral_en_ita_sgd_5k_1e4_hellaswag = [54.37, 59.26, 60.05, 60.77, 61.42, 62.69, 64.15, 64.33, 64.94, 64.79]
mistral_en_ita_sgd_5k_1e5_hellaswag = [59.66, 62.69, 63.07, 63.58, 63.98, 64.1, 64.57, 64.89, 65.06, 65.08]
original_performances = 61.1

plt.figure(figsize=(10,10))
plt.hlines(original_performances, xmin=0, xmax=len(mistral_ita_1e4_hellaswag)-1, label="original performances")
# plt.plot(list(range(len(mistral_ita_1e4_hellaswag))), mistral_ita_1e4_hellaswag, "-.", label="mistral italian lr = 1e4")
plt.plot(list(range(len(mistral_en_ita_1e4_hellaswag))), mistral_en_ita_1e4_hellaswag, "-.", label="mistral english italian lr = 1e4")
plt.plot(list(range(len(mistral_en_ita_sgd_5k_1e4_hellaswag))), mistral_en_ita_sgd_5k_1e4_hellaswag, "-.", label="mistral english italian adapted lr = 1e4")
plt.plot(list(range(len(mistral_en_ita_1e5_hellaswag))), mistral_en_ita_1e5_hellaswag, "--", label="mistral english italian lr = 1e5")
plt.plot(list(range(len(mistral_en_ita_sgd_5k_1e5_hellaswag))), mistral_en_ita_sgd_5k_1e5_hellaswag, "--", label="mistral english italian adapted lr = 1e5")
plt.legend()
plt.xticks(ticks=list(range(len(mistral_ita_1e4_hellaswag))), labels=["ba200", "ba400", "ba600", "ba800", "ba1000", "ba1200", "ba1400", "ba1600", "ba1800", "ba2000"], rotation=20)
plt.savefig("hellaswag_results.png")
plt.cla()

# MMLU IT ITALIAN
mistral_ita_1e4_mmlu_it = [25.99, 23.7, 26.33, 25.22, 26.86, 28.34, 27.8, 30.88, 30.53, 31.43]
mistral_en_ita_1e4_mmlu_it = [29.07, 28.62, 29.08, 28.69, 28.99, 30.63, 30.29, 32.22, 30.8, 30.99]
mistral_en_ita_1e5_mmlu_it = [45.85, 44.9, 45.52, 46.36, 45.83, 46.21, 46.21, 47.01, 47.29, 47.58]
mistral_en_ita_sgd_5k_1e4_mmlu_it = [25.54, 27.74, 31.68, 29.21, 31.43, 28.89, 30.35, 32.25, 33.32, 31.45]
mistral_en_ita_sgd_5k_1e5_mmlu_it = [40.18, 41.03, 42.4, 41.34, 42.43, 43.29, 43.93, 43.46, 43.08, 43.33]
original_performances = 46.5

plt.figure(figsize=(10,10))
plt.hlines(original_performances, xmin=0, xmax=len(mistral_ita_1e4_mmlu_it)-1, label="original performances")
# plt.plot(list(range(len(mistral_ita_1e4_mmlu_it))), mistral_ita_1e4_mmlu_it, "-.", label="mistral italian lr = 1e4")
plt.plot(list(range(len(mistral_en_ita_1e4_mmlu_it))), mistral_en_ita_1e4_mmlu_it, "-.", label="mistral english italian lr = 1e4")
plt.plot(list(range(len(mistral_en_ita_sgd_5k_1e4_mmlu_it))), mistral_en_ita_sgd_5k_1e4_mmlu_it, "-.", label="mistral english italian adapted lr = 1e4")
plt.plot(list(range(len(mistral_en_ita_1e5_mmlu_it))), mistral_en_ita_1e5_mmlu_it, "--", label="mistral english italian lr = 1e5")
plt.plot(list(range(len(mistral_en_ita_sgd_5k_1e5_mmlu_it))), mistral_en_ita_sgd_5k_1e5_mmlu_it, "--", label="mistral english italian adapted lr = 1e5")
plt.legend()
plt.xticks(ticks=list(range(len(mistral_ita_1e4_mmlu_it))), labels=["ba200", "ba400", "ba600", "ba800", "ba1000", "ba1200", "ba1400", "ba1600", "ba1800", "ba2000"], rotation=20)
plt.savefig("mmlu_it_results.png")
plt.cla()

# MMLU EN ENGLISH
mistral_ita_1e4_mmlu_en = [26.17, 3.59, 4.9, 5.74, 6.33, 8.67, 7.23, 9.12, 9.68, 30.1]
mistral_en_ita_1e4_mmlu_en = [31.41, 28.68, 29.54, 28.99, 30.41, 32.49, 32.01, 33.61, 32.26, 32.81]
mistral_en_ita_1e5_mmlu_en = [54.22, 53.19, 52.84, 53.51, 52.69, 52.74, 52.58, 53.88, 54.25, 54.03]
mistral_en_ita_sgd_5k_1e4_mmlu_en = [26.64, 27.7, 33.23, 30.92, 32.19, 31.71, 34.2, 34.7, 35.37, 33.66]
mistral_en_ita_sgd_5k_1e5_mmlu_en = [51.22, 50.35, 50.41, 48.82, 50.21, 50.19, 51.34, 51.2, 50.6, 50.85]
original_performances = 57.4

plt.figure(figsize=(10,10))
plt.hlines(original_performances, xmin=0, xmax=len(mistral_ita_1e4_mmlu_en)-1, label="original performances")
# plt.plot(list(range(len(mistral_ita_1e4_mmlu_en))), mistral_ita_1e4_mmlu_en, "-.", label="mistral italian lr = 1e4")
plt.plot(list(range(len(mistral_en_ita_1e4_mmlu_en))), mistral_en_ita_1e4_mmlu_en, "-.", label="mistral english italian lr = 1e4")
plt.plot(list(range(len(mistral_en_ita_sgd_5k_1e4_mmlu_en))), mistral_en_ita_sgd_5k_1e4_mmlu_en, "-.", label="mistral english italian adapted lr = 1e4")
plt.plot(list(range(len(mistral_en_ita_1e5_mmlu_en))), mistral_en_ita_1e5_mmlu_en, "--", label="mistral english italian lr = 1e5")
plt.plot(list(range(len(mistral_en_ita_sgd_5k_1e5_mmlu_en))), mistral_en_ita_sgd_5k_1e5_mmlu_en, "--", label="mistral english italian adapted lr = 1e5")
plt.legend()
plt.xticks(ticks=list(range(len(mistral_ita_1e4_mmlu_en))), labels=["ba200", "ba400", "ba600", "ba800", "ba1000", "ba1200", "ba1400", "ba1600", "ba1800", "ba2000"], rotation=20)
plt.savefig("mmlu_en_results.png")
plt.cla()