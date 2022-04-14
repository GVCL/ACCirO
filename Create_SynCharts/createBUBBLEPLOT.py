# libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib


cmap = matplotlib.cm.get_cmap('brg')
# cmap = matplotlib.cm.get_cmap('Dark2')
# cmap = matplotlib.cm.get_cmap('Accent')
# cmap = matplotlib.cm.get_cmap('hsv')
# cmap = matplotlib.cm.get_cmap('cool')
# cmap = matplotlib.cm.get_cmap('tab10')/



data = pd.read_csv("Downloads/Untitled-1.csv", sep=",", index_col=False)
data = data.loc[data.Year_Birth == 1989]
data = data[["NumWebPurchases", "NumWebVisitsMonth", "Education", "Income"]]
data["NumWebPurchases"] = data["NumWebPurchases"].apply(lambda x: x*10)
data["NumWebVisitsMonth"] = data["NumWebVisitsMonth"].apply(lambda x: x*10)
print(data)

# use the scatterplot function
fig, ax = plt.subplots(figsize=(10,8))
# fig, ax = plt.subplots(figsize=(8, 8))
num_of_colors   = len(data['Education'].unique())
colors=[cmap(i/num_of_colors) for i in range(num_of_colors)]
# colors=[cmap(i) for i in range(num_of_colors)][::-1]
axes = sns.scatterplot(data=data, x="NumWebPurchases", y="NumWebVisitsMonth", hue="Education", size="Income", palette=colors, alpha=0.75, sizes=(-20, -10))

# Legend split and place outside #
num_of_colors   +=  1
handles, labels = axes.get_legend_handles_labels()
color_hl = handles[:num_of_colors], labels[:num_of_colors]
sizes_hl = handles[num_of_colors:], labels[num_of_colors:]
# Call legend twice #
color_leg = axes.legend(*color_hl,
                        bbox_to_anchor = (1.05, 0.95),
                        loc            = 'upper left',
                        borderaxespad  = 0.,
                        labelspacing = 1.2)
for lh in color_leg.legendHandles:
    lh.set_alpha(0.75)
    lh._sizes = [100]
sizes_leg = axes.legend(*sizes_hl,
                        bbox_to_anchor = (1.05, 0.3),
                        borderaxespad  = 0.,
                        loc            = 'center left',
                        labelspacing = 2.2)

axes.add_artist(color_leg)
plt.subplots_adjust(right=0.75)
# plt.xlabel("Amount on Fruits")
# plt.ylabel("Amount on Meat")
plt.title("Purchase trends of people born in 1989")

# show the graph
plt.show()

# # Writing data to CSV file
# data.to_csv("SYNTHETIC_DATA/BUBBLE/data_bb2.csv", index = False)
# plt.savefig('SYNTHETIC_DATA/BUBBLE/bb2.png')
#





# data = pd.read_csv("Downloads/7.csv", sep=",", index_col=False)
# # data = data.loc[data.Time period Regions == "Australia"].append(data.loc[data.Result == "Canada"]).append(data.loc[data.Result == "United States of America"])
# data['Result'] = data['Result'].apply(lambda x: 'Loser' if x == 'L' else 'Winner')
# data = data[["Creep Score", "Champion Damage Share", "Result","Kills"]]
#
# data =data.head(80).tail(20)
# # data["Creep Score"] = data["Creep Score"].apply(lambda x: x*100)
# data["Champion Damage Share"] = data["Champion Damage Share"].apply(lambda x: x*100)
# # data["Kills"] = data["Kills"].apply(lambda x: x/1000000)
# # use the scatterplot function
#
# fig, ax = plt.subplots(figsize=(10,8))
# # fig, ax = plt.subplots()
# num_of_colors   = len(data["Result"].unique())
# colors=[cmap(i/num_of_colors) for i in range(num_of_colors)][::-1]
# axes = sns.scatterplot(data=data, x="Creep Score", y="Champion Damage Share", hue="Result", size="Kills",  palette=colors, alpha=0.5, sizes=(1000,1500))
# # Legend split and place outside #
# num_of_colors += 1
# handles, labels = axes.get_legend_handles_labels()
# color_hl = handles[:num_of_colors], labels[:num_of_colors]
# sizes_hl = handles[num_of_colors:], labels[num_of_colors:]
# # Call legend twice #
# color_leg = axes.legend(*color_hl,
#                         bbox_to_anchor = (1.05, 0.97),
#                         loc            = 'upper left',
#                         borderaxespad  = 0.,
#                         labelspacing = 1.0)
# for lh in color_leg.legendHandles:
#     lh.set_alpha(0.5)
#     lh._sizes = [100]
# sizes_leg = axes.legend(*sizes_hl,
#                         bbox_to_anchor = (1.05, 0.29),
#                         borderaxespad  = 0.,
#                         loc            = 'center left',
#                         labelspacing = 2.2)
#
# axes.add_artist(color_leg)
# plt.subplots_adjust(right=0.75)
# # plt.xlabel("Cloud")
# # plt.ylabel("Va")
# plt.title("League of Legends 2021 World Championship Game")
# # # #
# # plt.show()# show the graph
#
# # # Writing data to CSV file
# data.to_csv("SYNTHETIC_DATA/BUBBLE/data_bb14.csv", index = False)
# plt.savefig('SYNTHETIC_DATA/BUBBLE/bb14.png')






data = pd.read_csv("Downloads/forecast_data.csv", sep=",", index_col=False)
data = data.loc[data.city == "Chennai"]
data['rainfall'] = data['is_day'].apply(lambda x: 'Light rain' if x == 0 else 'Heavy rain')
data = data[["cloud", "temp_f", "rainfall","heatindex_f"]]

data =data.tail(80).head(25)
# data["Creep Score"] = data["Creep Score"].apply(lambda x: x*100)
# data["Champion Damage Share"] = data["Champion Damage Share"].apply(lambda x: x*100)
# data["Kills"] = data["Kills"].apply(lambda x: x/1000000)
# use the scatterplot function

fig, ax = plt.subplots(figsize=(10,8))
# fig, ax = plt.subplots()
num_of_colors   = len(data["rainfall"].unique())
colors=[cmap(i/num_of_colors) for i in range(num_of_colors)][::-1]
axes = sns.scatterplot(data=data, x="cloud", y="temp_f", hue="rainfall", size="heatindex_f",  palette=colors, alpha=0.5, sizes=(250,1000))
# Legend split and place outside #
num_of_colors += 1
handles, labels = axes.get_legend_handles_labels()
color_hl = handles[:num_of_colors], labels[:num_of_colors]
sizes_hl = handles[num_of_colors:], labels[num_of_colors:]
# Call legend twice #
color_leg = axes.legend(*color_hl,
                        bbox_to_anchor = (1.05, 0.97),
                        loc            = 'upper left',
                        borderaxespad  = 0.,
                        labelspacing = 1.0)
for lh in color_leg.legendHandles:
    lh.set_alpha(0.5)
    lh._sizes = [100]
sizes_leg = axes.legend(*sizes_hl,
                        bbox_to_anchor = (1.05, 0.29),
                        borderaxespad  = 0.,
                        loc            = 'center left',
                        labelspacing = 2.2)

axes.add_artist(color_leg)
plt.subplots_adjust(right=0.75)
plt.xlabel("Temperature")
plt.ylabel("Cloud")
plt.title("Rainfall condition in chennai")
# # #
# plt.show()# show the graph

# # Writing data to CSV file
data.to_csv("SYNTHETIC_DATA/BUBBLE/data_bb5.csv", index = False)
plt.savefig('SYNTHETIC_DATA/BUBBLE/bb5.png')
