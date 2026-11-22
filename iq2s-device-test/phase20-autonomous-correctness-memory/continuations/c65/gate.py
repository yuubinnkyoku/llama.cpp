import numpy as np, sys
n_vocab = 200058
steps = 32
cpu = np.fromfile(sys.argv[1], dtype=np.float32).reshape(steps, n_vocab)
htp = np.fromfile(sys.argv[2], dtype=np.float32).reshape(steps, n_vocab)
am_c = cpu.argmax(axis=1); am_h = htp.argmax(axis=1)
agree = int((am_c == am_h).sum())
print("argmax agreement", agree, "/", steps)
diff = htp - cpu
print("logit diff: mean_abs %.6f max_abs %.6f rms %.6f" % (np.abs(diff).mean(), np.abs(diff).max(), np.sqrt((diff**2).mean())))
# per-step correlation
cors = []
for i in range(steps):
    a, b = cpu[i], htp[i]
    cors.append(np.corrcoef(a, b)[0,1])
print("per-step corr min %.8f mean %.8f" % (min(cors), float(np.mean(cors))))
def logsoftmax(x):
    m = x.max()
    z = x - m
    return z - np.log(np.exp(z).sum())
nll_c = []; nll_h = []
for i in range(steps):
    t = int(am_c[i])
    nll_c.append(-logsoftmax(cpu[i])[t])
    nll_h.append(-logsoftmax(htp[i])[t])
nll_c = np.array(nll_c); nll_h = np.array(nll_h)
d = nll_h - nll_c
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
print("NLL cpu %.6f htp %.6f delta %.6f" % (nll_c.mean(), nll_h.mean(), d.mean()))
print("NLL delta t-stat %.4f" % t)
# teacher-forced agreement over the 32 generated tokens: compare argmax of the *next* step logits
print("teacher-forced tokens (cpu argmax):", " ".join(str(int(x)) for x in am_c[:8]), "...")
print("per-step NLL delta:", " ".join("%.4f" % v for v in d[:8]), "...")
# top-10 logit gap: for each step, is the cpu token still in htp top10?
top10 = 0
for i in range(steps):
    idx = np.argpartition(htp[i], -10)[-10:]
    if int(am_c[i]) in idx: top10 += 1
print("cpu argmax in htp top10:", top10, "/", steps)
