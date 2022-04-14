import numpy as np
import csv# v 1.19.2
import matplotlib.pyplot as plt    # v 3.3.2
import matplotlib
#
# Create random data


dot_params = []
rng = np.random.default_rng(270)
data = rng.integers(2015,2020, size=18)
values, cnt = np.unique(data, return_counts=True)
dot_params += [values]
dot_params += [cnt]

rng = np.random.default_rng(540)
data = rng.integers(2015,2020, size=18)
print(np.unique(data, return_counts=True))
_, cnt = np.unique(data, return_counts=True)
dot_params += [cnt]

rng = np.random.default_rng(175)
data = rng.integers(2015,2020, size=18)
print(np.unique(data, return_counts=True))
_, cnt = np.unique(data, return_counts=True)
dot_params += [cnt]

# rng = np.random.default_rng(360)
# data = rng.integers(2015,2020, size=18)
# print(np.unique(data, return_counts=True))
# _, cnt = np.unique(data, return_counts=True)
# dot_params += [cnt]



cmap = matplotlib.cm.get_cmap('Accent')
# cmap = matplotlib.cm.get_cmap('brg')
# cmap = matplotlib.cm.get_cmap('Dark2')
dot_params = np.array(dot_params)
num_of_colors   = len(dot_params) - 1
colors=[cmap(i*5) for i in range(num_of_colors)][::-1]
labels= ['X','Prime','Business','Standard']

# Draw dot plot with appropriate figure size, marker size and y-axis limits
fig, ax = plt.subplots(figsize=(6,4.8))
for i in range(1,len(dot_params)) :
    for j in range(len(dot_params[0])):
        k = np.sum(dot_params[1:i,j])+1
        ax.plot([dot_params[0][j]]*dot_params[i][j], list(range(k,dot_params[i][j]+k)), 'o', ms=9, color=colors[i-1], linestyle='')
    k = np.sum(dot_params[1:i,0])+1
    ax.plot([dot_params[0][0]]*dot_params[i][0], list(range(k,dot_params[i][0]+k)),'o', ms=9, color=colors[i-1], linestyle='', label=labels[i])
# ax.set_ticks(range(min(values), max(values)+1))
ax.tick_params(axis='x', length=0, pad=8, labelsize=12)
ax.tick_params(axis='y', length=0, pad=8, labelsize=12)
plt.subplots_adjust(right=0.75)

plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.xlabel("Year")
plt.ylabel("Sales (in million)")
plt.title("Order shipment sales per year")
# plt.show()

# # Writing data to CSV file
dot_params = [labels]+dot_params.transpose().tolist()
with open("SYNTHETIC_DATA/DOT/data_dp16.csv", 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerows(dot_params)
plt.savefig('SYNTHETIC_DATA/DOT/dp16.png')

# import numpy as np
# import csv# v 1.19.2
# import matplotlib.pyplot as plt    # v 3.3.2
#
# # Create random data
# rng = np.random.default_rng(87) # random number generator
# # data = rng.integers(1,12, size=40)
# data = rng.integers(2000,2007, size=25)
# values, counts = np.unique(data, return_counts=True)
#
# # Draw dot plot with appropriate figure size, marker size and y-axis limits
# fig, ax = plt.subplots(figsize=(6,4.8))#
# fig, ax = plt.subplots(figsize=(6,3))#
# for value, count in zip(values, counts):
#     print(value, count)
#     ax.plot([value]*count, list(range(1,count+1)), 'o', ms=9, color='#DE3163', linestyle='')
#
# # ax.set_xticks(range(min(values), max(values)+1))
# ax.tick_params(axis='x', length=0, pad=8, labelsize=12)
# ax.tick_params(axis='y', length=0, pad=8, labelsize=12)
# # plt.subplots_adjust(right=0.75)
#
# plt.xlabel("sample")
# plt.ylabel("no.of states")
# plt.title("States visited every year")
#
# plt.show()
#
# # # Writing data to CSV file
# # dot_params = zip(['X']+list(values), ['Y']+list(counts))
# # with open("SYNTHETIC_DATA/DOT/data_dp16.csv", 'w', newline='') as file:
# #     writer = csv.writer(file, delimiter=',')
# #     writer.writerows(dot_params)
# # plt.savefig('SYNTHETIC_DATA/DOT/dp16.png')
# #
