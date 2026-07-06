### Juego

def cal_verdad():
    print("Opción seleccionada valores de verdad")
    print("Ingresa V para verdadero o F para falso")
    p_input = input("p:")
    p_value = None
    if p_input == "V":
        p_value = True
    else:
        p_value = False
    q_input = input("q:")
    q_value = None
    if q_input == "V":
        q_value = True
    else:
        q_value = False
    
    op_input = input("Ingresa operador: C=Conjunción, D=Disyunción, I=Condicional, B=Bicondicional: ")
    if op_input == "C":
        res = p_value and q_value
        if res:
            print("V")
        else:
            print("F")
    elif op_input == "D":
        res = p_value or q_value
        if res:
            print("V")
        else:
            print("F")

def convertir_txt_verdad(txt):
    if txt == "V":
        return True
    else:
        return False
    
def cal_lenguaje():
    print("Opción seleccionada Lenguaje Natural")
    p_input = input("Ingresa la porposicion p: ")
    q_input = input("Ingresa la porposicion q: ")
    op_input = input("Ingresa operador: C=Conjunción, D=Disyunción, I=Condicional, B=Bicondicional: ")
    if op_input == "C":
        print(p_input, "y", q_input)
    elif op_input == "D":
        print(p_input, "o", q_input)

def calculadora():
    print("Opcion seleccionada calculadora:")

    opcion = input("Ingresa un modo: 1=Verdadero/False o 2=LenguajeNatural: ")
    if opcion == "1":
        cal_verdad()
    else:
        cal_lenguaje()

def evaluar_respuesta(op1, op2, op3, in_op1, in_op2, in_op3):
    """Funcion que compara las respuestas"""
    if op1 == in_op1 and op2==in_op2 and op3==in_op3:
        return True
    else:
        return False
        

def compute_res(var1, var2, var3, var4, op1, op2, op3):
    res1 = True
    res2 = True
    res_final = True

    var1 = convertir_txt_verdad(var1)
    var2 = convertir_txt_verdad(var2)
    var3 = convertir_txt_verdad(var3)
    var4 = convertir_txt_verdad(var4)

    if op1 == "C":
        res1 = var1 and var2
    elif op1 == "D":
        res1 = var1 or var2

    if op3== "C":
        res2 = var1 and var2
    elif op1 == "D":
        res2 = var1 or var2

    if op3== "C":
        res_final = res1 and res2
    elif op1 == "D":
        res_final = res1 or res2
    
    return res_final


def minijuego():
    print("Opcion seleccionada calculadora:")
    print("Tienes 4 posibles variables para utilizar (p,q,r,s)")
    print("Tines la siguiente estructura de proposición (var1 op1 var2) op2 (var3 op3 var4)")
    op1 = input("Ingresa operador1: C=Conjunción, D=Disyunción, I=Condicional, B=Bicondicional: ")
    op2 = input("Ingresa operador2: C=Conjunción, D=Disyunción, I=Condicional, B=Bicondicional: ")
    op3 = input("Ingresa operador3: C=Conjunción, D=Disyunción, I=Condicional, B=Bicondicional: ")

    print("Proposición formada: (var", op1, "var2)", op2, "(var3", op3, "var4)")

    print()
    print("El usuario 2 inicia aca!")
    intentos = 1
    flag = True
    while flag:
        print("Intento", intentos)
        var1 = input("Ingresa var1 (V o F):")
        var2 = input("Ingresa var2 (V o F):")
        var3 = input("Ingresa var3 (V o F):")
        var4 = input("Ingresa var4 (V o F):")
        resultado = compute_res(var1, var2, var3, var4, op1, op2, op3)
        if resultado:
            print("EL resultado es Verdadero")
        else:
            print("El resultado es Falso")
        print("Tienes la respuesta?")
        in_op1 = input("Ingresa el operador 1: ")
        in_op2 = input("Ingresa el operador 2: ")
        in_op3 = input("Ingresa el operador 3: ")

        resp_final = evaluar_respuesta(op1, op2, op3, in_op1, in_op2, in_op3)

        if resp_final:
            print("Respuesta Correcta :)")
            break
        else:
            print("Respuesta incorrecta :(")
            intentos += 1



def main():
    print("Menu")
    opcion = input("Ingresa un modo: 1=Calculadora de Verdad o 2=Minijuego: ")
    if opcion == "1":
        calculadora()
    else:
        minijuego()

if __name__ == "__main__":
    main()