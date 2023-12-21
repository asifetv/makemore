import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pdb
import random

#readin all the words
with open("names.txt", 'r') as f:
    words = f.read ().splitlines()
#print(words[:100])

chars = sorted(list(set(''.join(words))))
stoi = {c:i+1 for i,c in enumerate(chars)}
stoi['.'] = 0
itos = {i:c for c,i in stoi.items()}
#print(stoi,itos)

block_size = 3 #context length: how many characters do we take to predict the next one

def build_dataset(words):

    X, Y = [], []

    for w in words:
        #print(w)
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(stoi[ch])
            context = context[1:] + [ix]
            #print(''.join(itos[i] for i in context), '--------------->', itos[ix])

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    print(X.shape, Y.shape)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 2), generator=g, requires_grad=True)
w1 = torch.randn((6, 100), generator=g, requires_grad=True)
b1 = torch.randn(100, generator=g, requires_grad=True)
w2 = torch.randn((100, 27), generator=g, requires_grad=True)
b2 = torch.randn(27, generator=g, requires_grad=True)
parameters = [C, w1, b1, w2, b2]

number_parameter = sum(p.nelement() for p in parameters)
#print(number_parameter)

lre = torch.linspace(-3, 0, 1000)
lrs = 10**lre

lri = []
lossi = []
for i in range(30000):
    #minibatch
    ix = torch.randint(0, Xtr.shape[0], (32,))
    #forward pass
    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view(-1, 6) @ w1 + b1)
    logits = h @ w2 + b2
    #counts = logits.exp()
    #prob = counts/counts.sum(1, keepdim=True)
    #loss = -prob[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[ix])

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    #Update parameter
    for p in parameters:
        #p.data += -lrs[i] * p.grad
        p.data += -0.1 * p.grad
    
    #Track stats
    #lri.append(lrs[i].item())
    #lossi.append(loss.item())
 
#print("plotting")
#plt.plot(lri, lossi)
#plt.show()
print ("Training loss --", loss.item())

emb = C[Xdev]
h = torch.tanh(emb.view(-1, 6) @ w1 + b1)
logits = h @ w2 + b2
loss = F.cross_entropy(logits, Ydev)
print("Dev loss -- ", loss.item())

emb = C[Xtr]
h = torch.tanh(emb.view(-1, 6) @ w1 + b1)
logits = h @ w2 + b2
loss = F.cross_entropy(logits, Ytr)
print("Dev loss -- ", loss.item())

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ w1 + b1)
        logits = h @ w2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
    print (''.join(itos[i] for i in out))


plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i, 1].item(), itos[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()

pdb.set_trace()



