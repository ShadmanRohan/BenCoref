import json
import sys
import collections
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


def mention_clusters_length(df, count):
  for i in range(len(df['mention_clusters'])):
    for j in range(len(df['mention_clusters'][i])):
      length = len(df['mention_clusters'][i][j])
      count[length] += 1

def sentences_length(df, count):
  for i in range(len(df['sentences'])):
    length = 0
    for j in range(len(df['sentences'][i])):
      length += len(df['sentences'][i][j])
    count[length] += 1

def total_token_count(df):
  tokens = 0
  for i in range(len(df)):
    for j in range(len(df['sentences'][i])):
      tokens += len(df['sentences'][i][j])
  return tokens

def total_mention_token_count(d):
  total_mention_token = 0
  for key, value in d.items():
    total_mention_token += key*value
  return total_mention_token

def mention_len_count(df):
  mention_len = {}
  for i in range(len(df)):
    for j in range(len(df['mention_clusters'][i])):
      length = len(df['mention_clusters'][i][j])
      if length != 1:
        for k in range(length):
          l = df['mention_clusters'][i][j][k][2]-df['mention_clusters'][i][j][k][1]
          if l not in mention_len.keys():
            mention_len[l] = 1
          else:
            mention_len[l] += 1
  return total_mention_token_count(mention_len)

def token_positions(text, cluster):
  sentence_len = list(map(len, text))
  cluster = [list(x) for x in set(tuple(x) for x in cluster)]
  positions = []
  for i in range(len(cluster)):
    if cluster[i][0] == 0:
      positions.append(cluster[i][1])
    else:
      temp = sum(sentence_len[0:cluster[i][0]]) + cluster[i][1]
      positions.append(temp)
  return sorted(positions)

def cluster_distance(text, cluster):
  positions = token_positions(text, cluster)
  separation_dis = []
  for i in range(len(positions)-1):
    separation_dis.append(np.abs(positions[i] - positions[i+1]))
  return separation_dis

def standard(mean, data):
  s = 0
  for d,v in data.items():
    tem = abs(d-mean)*v
    s += v**2
  return s/sum(data.values())

def distance_category(df):
  distance_count = Counter()
  for i in range(len(df)):
    text = df['sentences'][i]
    clusters = df['mention_clusters'][i]
    for j in range(len(clusters)):
      dist = cluster_distance(text, clusters[j])
      for k in range(len(dist)):
        if dist[k] not in distance_count.keys():
          distance_count[dist[k]] = 1
        else:
          distance_count[dist[k]] += 1
  return distance_count

def print_mean_std(df, category_name):
  distance_count = distance_category(df)
  total_dis = 0
  total_mention = 0
  for key, value in distance_count.items():
    total_dis += key*value
    total_mention += value

  mean = total_dis/total_mention

  print(category_name)
  print("Mean: {:.2f}".format(mean))
  print("Std: {:.2f}\n".format(np.sqrt(standard(mean, distance_count))))

def calculate_cluster_percentage(d, clusters):
  percentage = {16:0}
  for key, value in d.items():
    if key < 16:
      percentage[key] = value
    else:
      percentage[16] += value
  if 1 in percentage.keys():
    del percentage[1]
  for key, value in percentage.items():
    percentage[key] = (value*100.0)/clusters
  return percentage

def distribution_per_category(df, d, category_type):
  for i in range(len(df)):
    for k in range(len(df['sentences'][i])):
      d[category_type]['Tokens'] += len(df['sentences'][i][k])
    d[category_type]['Clusters'] += len(df['mention_clusters'][i])
    for j in range(len(df['mention_clusters'][i])):
      d[category_type]['Mentions'] += len(df['mention_clusters'][i][j])
  return d

def total_tokens(text):
  length = 0
  for i in range(len(text)):
    length += len(text[i])
  return length

def distance_between_first_and_last(cluster, text):
  cluster = sorted(cluster)
  first_mention = cluster[0]
  last_mention = cluster[-1]
  length = total_tokens(text[first_mention[0]:last_mention[0]])
  dis = length + last_mention[1] - first_mention[1]
  return dis

def spread(df):
  d = {}
  for i in range(len(df)):
    clusters = df['mention_clusters'][i]
    text = df['sentences'][i]
    for j in range(len(clusters)):
      spread = distance_between_first_and_last(clusters[j], text)
      if spread in d.keys():
        d[spread] += 1
      else:
        d[spread] = 1
  return collections.OrderedDict(sorted(d.items()))

def count(d, key):
  l = [0]
  for i in range(1,len(key)):
    if i == len(key)-1:
      t = [x for x in d.keys() if x>key[i]]
    else:
      t = [x for x in d.keys() if x>key[i-1] and x<=key[i]]
    total = 0
    for k in t:
      total += d[k]
    l.append(total)
  return l

if __name__ == "__main__":
	bencoref_path = sys.argv[1]
	preco_path = sys.argv[2]
	data = []
	print("loading PreCo......")
	with open((preco_path+'/train.jsonl')) as f:
	    for line in f:
	        data.append(json.loads(line))

	df_preco = pd.DataFrame(data)

	print("loading BenCoref......\n")
	story_df = pd.read_json((bencoref_path+'/story.json'), encoding='utf8')
	novel_df = pd.read_json((bencoref_path+'/novel.json'), encoding='utf8')
	biography_df = pd.read_json((bencoref_path+'/biography.json'), encoding='utf8')
	descriptive_df = pd.read_json((bencoref_path+'/descriptive.json'), encoding='utf8')
	df_bencoref = pd.concat([biography_df, descriptive_df, novel_df, story_df])
	df_bencoref.reset_index(drop=True, inplace=True)
	df_bencoref = df_bencoref.set_index('id')

	# size and property comparison
	df_bencoref_clusters = Counter()
	df_preco_clusters = Counter()
	df_bencoref_doc = Counter()
	df_preco_doc = Counter()

	mention_clusters_length(df_bencoref, df_bencoref_clusters)
	mention_clusters_length(df_preco, df_preco_clusters)
	sentences_length(df_bencoref, df_bencoref_doc)
	sentences_length(df_preco, df_preco_doc)
	print("Number Of Texts: \nPreCo Dataset: {}\nBenCoref Dataset: {}\n".format(len(df_preco), len(df_bencoref)))
	print("Text Length (Number token in a text): \nPreCo Dataset: Max-> {}, Min-> {}".format(max(df_preco_doc), min(df_preco_doc)))
	print("BenCoref Dataset: Max-> {}, Min-> {}\n".format(max(df_bencoref_doc), min(df_bencoref_doc)))
	print("Mention Clusters Length: \nPreco Dataset: Max-> {}, Min-> {}".format(max(df_preco_clusters), min(df_preco_clusters)))
	print("BenCoref Dataset: Max-> {}, Min-> {}\n".format(max(df_bencoref_clusters), min(df_bencoref_clusters)))

	total_tag_bencoref = sum(np.fromiter(df_bencoref_clusters.keys(), dtype=int)*np.fromiter(df_bencoref_clusters.values(), dtype=int))
	total_tag_preco = sum(np.fromiter(df_preco_clusters.keys(), dtype=int)*np.fromiter(df_preco_clusters.values(), dtype=int))
	print("Total Mention Clusters: \nPreCo Dataset: {}\nBenCoref Dataset: {}\n".format(sum(
	    df_preco_clusters.values()), sum(df_bencoref_clusters.values())))
	print("Total Mentions: \nPreCo Dataset: {}\nBenCoref Dataset: {}\n".format(total_tag_preco, total_tag_bencoref))

	bn_tokens = total_token_count(df_bencoref)
	pre_tokens = total_token_count(df_preco)

	print("total tokens PreCo: {}".format(pre_tokens))
	print("total tokens BenCoref: {}\n".format(bn_tokens))

	without_singleton = total_tag_preco - df_preco_clusters[1]
	cluster_without = sum(df_preco_clusters.values())-df_preco_clusters[1]
	total_mention_token_pre = mention_len_count(df_preco)
	total_mention_token_bn = mention_len_count(df_bencoref)

	print("Without Singleton cluster:")
	print("total mention (PreCo): {}".format(without_singleton))
	print("total cluster (PreCo): {}".format(cluster_without))
	print("total token in mention PreCo: {}".format(total_mention_token_pre))
	print("total token in mention BenCoref: {}\n".format(total_mention_token_bn))

	print_mean_std(biography_df, "Biography")
	print_mean_std(descriptive_df, "Descriptive")
	print_mean_std(story_df, "Story")
	print_mean_std(novel_df, "Novel")

	bn_cluster = sum(df_bencoref_clusters.values())
	bn_percentage = calculate_cluster_percentage(df_bencoref_clusters, bn_cluster)
	pre_percentage = calculate_cluster_percentage(df_preco_clusters, cluster_without)

	label = ['Bengali']*len(bn_percentage)
	label2 = ['PreCo']*len(pre_percentage)
	bn_df = pd.DataFrame.from_dict(data={'Cluster size':list(bn_percentage.keys()),
	                                     '%':list(bn_percentage.values()),
	                                     'label':label})
	pre_df = pd.DataFrame.from_dict(data={'Cluster size':list(pre_percentage.keys()),
	                                     '%':list(pre_percentage.values()),
	                                     'label':label2})
	df = pd.concat([bn_df, pre_df])
	df.reset_index(drop=True, inplace=True)

	sns.set_theme()
	palette = sns.color_palette("RdGy", 5)
	ax =sns.barplot(x='Cluster size', y='%', hue='label', data = df, palette=palette)
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles=handles[0:], labels=labels[0:])
	la = ax.get_xticklabels()
	la[-1] = '16+'
	ax.set_xticklabels(la)
	fig = ax.get_figure()
	fig.savefig("cluster_percentage.png")
	plt.clf()
	plt.cla()
	plt.close()

	data_dict = {"Descriptive":
	             {"Clusters": 0, "Mentions": 0, "Tokens": 0},
	             "Biography":
	             {"Clusters": 0, "Mentions": 0, "Tokens": 0},
	             "Story":
	             {"Clusters": 0, "Mentions": 0, "Tokens": 0},
	             "Novel":
	             {"Clusters": 0, "Mentions": 0, "Tokens": 0}
	            }
	data_dict = distribution_per_category(descriptive_df, data_dict, 'Descriptive')
	data_dict = distribution_per_category(biography_df, data_dict, 'Biography')
	data_dict = distribution_per_category(story_df, data_dict, 'Story')
	data_dict = distribution_per_category(novel_df, data_dict, 'Novel')

	for key, value in data_dict.items():
	  data_dict[key] = {"Clusters": ((data_dict[key]['Clusters']*100.0)/bn_cluster),
	                    "Mentions": ((data_dict[key]['Mentions']*100.0)/total_tag_bencoref),
	                    "Tokens": ((data_dict[key]['Tokens']*100.0)/bn_tokens)
	                   }

	percen = pd.DataFrame.from_dict(data_dict).T
	percen = percen.reset_index()
	mentions = percen.loc[:,['index', 'Mentions']]
	clusters = percen.loc[:,['index', 'Clusters']]
	tokens = percen.loc[:,['index', 'Tokens']]
	del percen

	label = ['Cluster']*4
	label2 = ['Mention']*4
	label3 = ['Token']*4
	clusters = pd.concat([clusters, pd.DataFrame(label)], axis=1)
	mentions = pd.concat([mentions, pd.DataFrame(label2)], axis=1)
	tokens = pd.concat([tokens, pd.DataFrame(label3)], axis=1)

	mentions.rename(columns={'Mentions': '%', 'index':'Categories'}, inplace = True)
	clusters.rename(columns={'Clusters': '%', 'index':'Categories'}, inplace = True)
	tokens.rename(columns={'Tokens': '%', 'index':'Categories'}, inplace = True)
	df = pd.concat([clusters, mentions, tokens])

	palette = sns.color_palette("PRGn", 8)
	ax =sns.barplot(x='Categories', y='%', hue=0, data = df, palette=palette)
	fig = ax.get_figure()
	fig.savefig("distribution.png")
	plt.clf()
	plt.cla()
	plt.close()

	temp = {}
	temp['all'] = spread(df_bencoref)

	x_val = [*range(0,1001,50)]
	bio = count(temp['all'], x_val)
	drop = []
	for i in range(len(x_val)):
	  if bio[i]==0:
	    drop.append(i)
	for i in drop[::-1]:
	  del x_val[i]
	  del bio[i]
	df = pd.DataFrame([x_val, bio]).T
	df.rename(columns={0:'Spread', 1:'Count'}, inplace=True)
	ax =sns.barplot(x='Spread', y='Count',data = df,color='#feb24c')
	ax.xaxis.set_major_locator(plt.MaxNLocator(6))
	ax.set_xticklabels([0,50,250,450,650,850])
	fig = ax.get_figure()
	fig.savefig("spread.png")
	plt.clf()
	plt.cla()
	plt.close()

	word_count = Counter()
	for i in range(len(df_bencoref)):
	  text = df_bencoref['sentences'][i]
	  mention_clusters = df_bencoref['mention_clusters'][i]
	  for j in range(len(mention_clusters)):
	    for k in range(len(mention_clusters[j])):
	      token_position = mention_clusters[j][k]
	      token = ' '.join(text[token_position[0]][token_position[1]:token_position[2]]).strip()
	      if token not in word_count.keys():
	        word_count[token] = 1
	      else:
	        word_count[token] += 1

	frequency = np.array(word_count.most_common(10))[:,1].astype(np.int64)
	tokens = np.array(word_count.most_common(10))[:,0]
	token_df = pd.DataFrame({"Tokens": tokens, "Frequency": frequency})
	sns.set(font='Nirmala UI')
	ax =sns.barplot(x='Tokens', y='Frequency', data = token_df, color='#31a354')
	fig = ax.get_figure()
	fig.savefig("token_frequency.png")
	plt.clf()
	plt.cla()
	plt.close()

	print("Figures Saved!!")
