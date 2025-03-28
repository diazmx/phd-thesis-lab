
A = set()
B = {1, 2, 3, 4}
C = {1, 2, 3, 5}
print(B, C)
print(B.union(C))
print(B.intersection(C))
print(B.difference(C))
print(B.issubset(C))

def subset(A, B):
    for x in A:
        bandera = False
        for i in B:
            if x == i:
                bandera = True
        if not bandera:
            print("No es subconjunto")
            return
    print("Es subconjunto")

print(subset(B, C))