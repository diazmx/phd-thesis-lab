class Funcion:
    def __init__(self):
        self.X = set()
        self.Y = set()
        self.f = {}

    def agregar_elemento_X(self, x):
        self.X.add(x)
        self.evaluar_propiedades()

    def quitar_elemento_X(self, x):
        self.X.discard(x)
        self.f.pop(x, None)
        self.evaluar_propiedades()

    def agregar_elemento_Y(self, y):
        self.Y.add(y)
        self.evaluar_propiedades()

    def quitar_elemento_Y(self, y):
        self.Y.discard(y)
        self.f = {x: v for x, v in self.f.items() if v != y}
        self.evaluar_propiedades()

    def agregar_tupla(self, x, y):
        if x in self.X and y in self.Y:
            self.f[x] = y
        else:
            print("Error: Asegúrese de que x está en X e y está en Y.")
        self.evaluar_propiedades()

    def quitar_tupla(self, x):
        self.f.pop(x, None)
        self.evaluar_propiedades()

    def es_inyectiva(self):
        return len(set(self.f.values())) == len(self.f.values())

    def es_sobreyectiva(self):
        return set(self.f.values()) == self.Y
    
    def es_funcion_valida(self):
        return set(self.f.keys()) == self.X and len(self.f.keys()) == len(set(self.f.keys()))

    def es_biyectiva(self):
        return self.es_inyectiva() and self.es_sobreyectiva()

    def evaluar_propiedades(self):
        print("\nEstado actual:")
        print(f"X: {self.X}")
        print(f"Y: {self.Y}")
        print(f"Función: {self.f}")
        print(f"Función válida: {self.es_funcion_valida()}")
        if self.es_funcion_valida():
            print(f"Inyectiva: {self.es_inyectiva()}")
            print(f"Sobreyectiva: {self.es_sobreyectiva()}")
            print(f"Biyectiva: {self.es_biyectiva()}")
        

# Ejemplo de interacción
f = Funcion()
"""f.agregar_elemento_X(1)
f.agregar_elemento_X(2)
f.agregar_elemento_Y('a')
f.agregar_elemento_Y('b')
f.agregar_tupla(1, 'a')
f.agregar_tupla(2, 'b')"""

while True:
    input_user = input("Selecciona una accion: 1) Agregar elemento, 2) Quitar Elemento: ")
    input_user2 = input("Selecciona a que conjunto: 1) Emisor, 2) Receptor, 3) Función: ")

    if input_user == "1" and input_user2 == "1":
        to_add = input("Ingresa elemento para agregar: ")
        f.agregar_elemento_X(to_add)
    elif input_user == "2" and input_user2 == "1":
        to_add = input("Ingresa elemento para quitar: ")
        f.quitar_elemento_X(to_add)
    elif input_user == "1" and input_user2 == "2":
        to_add = input("Ingresa elemento para agregar: ")
        f.agregar_elemento_Y(to_add)
    elif input_user == "1" and input_user2 == "2":
        to_add = input("Ingresa elemento para quitar: ")
        f.quitar_elemento_Y(to_add)
    elif input_user == "1" and input_user2 == "3":
        el1 = input("Ingresa elemento 1 de tupla para agregar: ")
        el2 = input("Ingresa elemento 2 de tupla para agregar: ")
        f.agregar_tupla(el1,el2)
    elif input_user == "1" and input_user2 == "3":
        el1 = input("Ingresa elemento 1 de tupla para quitar: ")
        el2 = input("Ingresa elemento 2 de tupla para quitar: ")
        f.quitar_tupla(el1,el2)