import pandas as pd
import pywt
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,7))
df = pd.read_csv("data/dataset_min.csv")
signal = df.iloc[0].to_numpy()
start_index = list(df.columns).index("400")
signal = signal[start_index:]
s = [signal[i] for i in range(len(signal)) if i%20 == 0]
print(s)

current_ticks = [i for i in range(210+1) if i%105 == 0]
custom_ticks = [c*20 for c in current_ticks]
plt.xticks(current_ticks, custom_ticks, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Discrete Band Indices", fontsize=20)
plt.ylabel("Reflectance", fontsize=20)
plt.plot(s,'o', markersize=1)
plt.title('Discrete Band Values', fontsize=20)
plt.show()
plt.figure(figsize=(5,7))
x = np.linspace(0,1,len(signal))
plt.plot(x,signal,markersize=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks([0.0,0.5,1.0])
plt.xlabel("Transformed Continuous Indices", fontsize=20)
plt.ylabel("Reflectance", fontsize=20)
plt.title('Continuous Function', fontsize=20)
plt.show()

# signal,_,_,_,_,_,_ = pywt.wavedec(signal, 'db1', level=6)
# print(signal.shape)
# plt.plot(signal)
# plt.show()

