
Emisores = {"A", "B", "C", "D"}
Receptores = {"C", "D", "E", "F"}
Llaves = {"k1", "k2", "k3"}

Conexiones = [
    ("A", "k1", "C"),
    ("B", "k2", "D"),
    ("C", "k3", "E"),
    ("D", "k1", "F")
]

































# Verificación de propiedades de funciones
def es_funcion(conexiones):
    emisores = set()
    for emisor, _, receptor in conexiones:
        if emisor in emisores:
            return False  # Un emisor está conectado a más de un receptor
        emisores.add(emisor)
    return True

def es_inyectiva(conexiones):
    receptores = set()
    for _, _, receptor in conexiones:
        if receptor in receptores:
            return False  # Dos emisores apuntan al mismo receptor
        receptores.add(receptor)
    return True

def es_sobreyectiva(conexiones, receptores):
    receptores_vistos = {r for _, _, r in conexiones}
    return receptores_vistos == receptores

def es_biyectiva(conexiones, receptores):
    return es_funcion(conexiones) and es_inyectiva(conexiones) and es_sobreyectiva(conexiones, receptores)

print("Verificación de propiedades de funciones:")
print(" - Es función:", es_funcion(Conexiones))
print(" - Inyectiva:", es_inyectiva(Conexiones))
print(" - Sobreyectiva:", es_sobreyectiva(Conexiones, Receptores))
print(" - Biyectiva:", es_biyectiva(Conexiones, Receptores))

# Verificación de propiedades de relaciones
def es_reflexiva(conexiones, conjunto):
    return all(any(a == x and b == x for a, _, b in conexiones) for x in conjunto)

def es_simetrica(conexiones):
    return all((b, k, a) in conexiones for a, k, b in conexiones)

def es_transitiva(conexiones):
    conexiones_dict = { (a, k): b for a, k, b in conexiones }
    for (a, k1), b in conexiones_dict.items():
        for (c, k2), d in conexiones_dict.items():
            if b == c and (a, k1, d) not in conexiones:
                return False
    return True

print("Verificación de propiedades de relaciones:")
print(" - Reflexiva:", es_reflexiva(Conexiones, Emisores))
print(" - Simétrica:", es_simetrica(Conexiones))
print(" - Transitiva:", es_transitiva(Conexiones))