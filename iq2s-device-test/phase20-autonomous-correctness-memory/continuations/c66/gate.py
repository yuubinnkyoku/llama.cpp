import numpy as np, sys
n_vocab = 200058
cpu = np.fromfile(sys.argv[1], dtype=np.float32)
htp = np.fromfile(sys.argv[2], dtype=np.float32)
assert cpu.size == htp.size and cpu.size % n_vocab == 0, (cpu.size, htp.size)
steps = cpu.size // n_vocab
cpu = cpu.reshape(steps, n_vocab); htp = htp.reshape(steps, n_vocab)
am_c = cpu.argmax(axis=1); am_h = htp.argmax(axis=1)
agree = int((am_c == am_h).sum())
print("steps", steps, "argmax agreement", agree, "/", steps)
diff = htp - cpu
print("logit diff: mean_abs %.6f max_abs %.6f rms %.6f" % (np.abs(diff).mean(), np.abs(diff).max(), np.sqrt((diff**2).mean())))
cors = [np.corrcoef(cpu[i], htp[i])[0,1] for i in range(steps)]
print("per-step corr min %.8f mean %.8f" % (min(cors), float(np.mean(cors))))
def logsoftmax(x):
    m = x.max(); z = x - m; return z - np.log(np.exp(z).sum())
nll_c = np.array([-logsoftmax(cpu[i])[int(am_c[i])] for i in range(steps)])
nll_h = np.array([-logsoftmax(htp[i])[int(am_c[i])] for i in range(steps)])
d = nll_h - nll_c
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print("NLL cpu %.6f htp %.6f delta %.6f" % (nll_c.mean(), nll_h.mean(), d.mean()))
print("NLL delta t-stat %.4f" % t)
top10 = 0
for i in range(steps):
    idx = np.argpartition(htp[i], -10)[-10:]
    if int(am_c[i]) in idx: top10 += 1
print("cpu argmax in htp top10:", top10, "/", steps)