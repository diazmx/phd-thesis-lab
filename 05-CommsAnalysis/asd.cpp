#include <iostream>

using namespace std;

class CuentaBancaria
{
private:
    string nombre;
    float saldo;

public:
    CuentaBancaria(string nombre_in, float saldo_in)
    {
        nombre = nombre_in;
        saldo = saldo_in;
    }

    void SetSaldo(float saldo)
    {
        saldo = saldo;
    }

    string GetNombre() const
    {
        return nombre;
    }

    void mostrar_saldo()
    {
        cout << "El cliente " << nombre << " tiene " << saldo << " saldo." << endl;
    }

    void depositar(float deposito)
    {
        saldo = saldo + deposito;
    }

    void retirar(float retiro)
    {
        if (retiro > saldo)
        {
            cout << "No tienes dinero suficiente" << endl;
        }
        else
        {
            saldo = saldo - retiro;
        }
    }
};

int main()
{

    CuentaBancaria cb1("Daniel", 1000);
    cb1.mostrar_saldo();
    cb1.depositar(500);
    cb1.mostrar_saldo();
    cb1.retirar(1500);
    cb1.mostrar_saldo();

    // cout << cb1.nombre;
    // cout << cb1.get_nombre();

    return 0;
}