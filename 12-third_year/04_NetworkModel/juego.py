import re
import time

def evaluar_proposicion(proposicion, valores):
    # Reemplazar las variables p, q, r, s con sus valores respectivos
    for variable, valor in valores.items():
        proposicion = proposicion.replace(variable, str(valor))

    # Transformar los operadores para que sean compatibles con Python
    proposicion = (proposicion
                   .replace("^", " and ")
                   .replace("v", " or ")
                   .replace("¬", " not ")
                   .replace("&&", " and ")
                   .replace("||", " or ")
                   .replace("!", " not "))

    # Evaluar la proposición
    try:
        return eval(proposicion)
    except Exception as e:
        print(f"Error en la evaluación de la proposición: {e}")
        return None

def main():
    print("Bienvenido al minijuego 'Hackeo de la proposición'\n")
    print("Usuario 1: Ingresa una proposición compuesta (usa p, q, r, s y operadores lógicos como ^, v, ¬)")
    proposicion = input("Proposición: ").strip()

    # Validar si la proposición tiene las variables correctas
    if not re.match(r"^[pqrs^v¬l ()!&|]+$", proposicion):
        print("Error: La proposición contiene símbolos inválidos.")
        return

    print("\nProposición registrada exitosamente.")
    print("Sistema encriptado listo para ser descifrado.\n")

    movimientos = 0
    tiempo_inicio = time.time()

    while True:
        print("Usuario 2: Ingresa los valores de verdad de las variables (0 o 1)")
        try:
            p = int(input("p: "))
            q = int(input("q: "))
            r = int(input("r: "))
            s = int(input("s: "))
        except ValueError:
            print("Por favor ingresa solo valores 0 o 1.")
            continue

        # Validar que solo se ingresen 0 o 1
        if not all(v in [0, 1] for v in [p, q, r, s]):
            print("Los valores de las variables deben ser 0 o 1.")
            continue

        # Evaluar proposición con los valores dados
        valores = {'p': p, 'q': q, 'r': r, 's': s}
        resultado = evaluar_proposicion(proposicion, valores)
        if resultado is None:
            break

        movimientos += 1
        print(f"Resultado: {resultado} (Intento {movimientos})\n")

        # Verificar si el usuario desea seguir jugando
        continuar = input("¿Has descifrado la proposición? (s/n): ").strip().lower()
        if continuar == 's':
            tiempo_total = time.time() - tiempo_inicio
            print("\nAcceso concedido.")
            print(f"Tiempo total: {tiempo_total:.2f} segundos.")
            print(f"Movimientos realizados: {movimientos}\n")
            break
        elif continuar != 'n':
            print("Entrada no válida. Continúa el juego.")

if __name__ == "__main__":
    main()
