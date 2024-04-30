import json

files = [
    'exp1_results/baseline.json',
    'exp1_results/go_baseline.json',
    'exp1_results/simple_morl.json',
    'exp1_results/morl_interepisode.json'
]

names = [
    'Goal Only',
    'Baseline',
    'Simple MORL',
    'MORL Inter-episode'
]

data = [
    json.load(open(file)) for file in files
]

table = {
    'completion_rate': [],
    'mean_reward': [],
    'total_steps': []
}

for name, d in zip(names, data):
    table['completion_rate'].append(f"{d['EOT']['completed_episodes'] / d['EOT']['total_episodes']:.2f}")
    table['mean_reward'].append(f"{d['EOT']['cummulative_rewards']:.2f}")
    table['total_steps'].append(f"{d['EOT']['total_steps']:.2f}")


table_str = '''
\\begin{table}
\\vspace{-.5in}
\\small
\\centering
\\caption{Table 1}
\\begin{tabular}{c|ccc|ccc|ccc|ccc}
\\toprule
& \\multicolumn{3}{c}{G.O.} & \\multicolumn{3}{c}{Baseline} & \\multicolumn{3}{c}{Simple MORL} & \\multicolumn{3}{c}{MORL Inter-episode} \\\\
\\midrule
& CR & MR & TS & CR & MR & TS & CR & MR & TS & CR & MR & TS \\\\
\\midrule
4x4 
'''.strip()

for i in range(len(names)):
    table_str += f"& {table['completion_rate'][i]} & {table['mean_reward'][i]} & {table['total_steps'][i]}"

# table_str = table_str[0:-1]

table_str += '''\\\\
\\bottomrule
\\end{tabular}
\\label{tab:exp1}
\\end{table}'''

print(table_str)


## total reward over training time
#
import matplotlib.pyplot as plt
import numpy as np

indices = [int(k) for k in data[0].keys() if k != 'EOT']
indices = [str(x) for x in sorted(indices, key=lambda x: int(x))]

for name, d in zip(names, data):
    plt.plot(indices, [d[x]['cummulative_rewards'] for x in indices], label=name)

plt.legend()
plt.xlabel('Episodes')
if len(indices) >10:
    plt.xticks(indices[::2])

plt.ylabel('Total Reward')
plt.title('Total Reward over Training Time')
plt.show()


# plt.figure()
# for name, d in zip(names, data):
#     plt.plot(np.arange(len(x['total_rewards'] for x in d if x != 'EOT')), d['EOT']['total_rewards'], lines[name], label=name)