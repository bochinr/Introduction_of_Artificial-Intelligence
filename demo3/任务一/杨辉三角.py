def triangles():
    N=[1]
    while True:
        yield N
        S=N[:]
        S.append(0)
        N=[S[i-1]+S[i] for i in range(len(S))]
n = 0
results = []
for t in triangles():
    print(t)
    results.append(t)
    n = n + 1
    if n == 10:
        break