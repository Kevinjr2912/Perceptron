from Perceptron import Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(path):
    data = pd.read_csv(path)
    x = data.iloc[:, [0,1]].to_numpy() 
    y_o = data.iloc[:, 2].to_numpy()
    return x, y_o

# Entrenar por lambda
def train_with_lambda(x, y_o, learning_rate, K):
    w = np.random.uniform(-20, 20, size=x.shape[1] + 1)
    perceptron = Perceptron(x, w, y_o)

    errors_per_epoch = []
    weights_per_epoch = []

    for k in range(K):
        weights_per_epoch.append(perceptron.weights.copy())

        # Yc = FU(XW)
        y_calculated = [perceptron.activation(yc) for yc in perceptron.calculate_scalar_product()]

        # E = Yc - Yo
        e = perceptron.loss(y_calculated)

        # ΔW = -λ X^T E
        adjust_weights = perceptron.adjust_weights(e, learning_rate)

        # W = W + ΔW
        perceptron.weights = perceptron.update_weights(adjust_weights)

        # Error = raiz(e1**2, e2**2, e3**2, ..., en**2)
        error_k = np.linalg.norm(e)
        errors_per_epoch.append(error_k)

    return np.array(errors_per_epoch), np.array(weights_per_epoch)

# Graficar pesos por lambda
def plot_weights(weights_history, lr):
    plt.figure(figsize=(8,5))
    for i in range(weights_history.shape[1]):
        plt.plot(range(weights_history.shape[0]), weights_history[:, i], label=f"w{i}")
    plt.xlabel("Época")
    plt.ylabel("Valor del peso")
    plt.title(f"Evolución de pesos (λ = {lr})")
    plt.legend()
    plt.grid(True)
    plt.show()

# Graficar errores globales
def plot_errors(all_errors, K):
    plt.figure(figsize=(10,6))
    for lr, errors in all_errors:
        plt.plot(range(K), errors, label=f"λ = {lr}")
    plt.xlabel("Época")
    plt.ylabel("Error (Norma L2)")
    plt.title("Error vs Época para distintos valores de λ")
    plt.legend()
    plt.grid(True)
    plt.show()

# Guardar pesos iniciales y finales
def save_weights_summary(weights_summary, filename="initial_final_weights.csv"):
    df_weights = pd.DataFrame(weights_summary)
    df_weights.to_csv(filename, index=False)
    print(f"Dataset de pesos guardado")

def main():
    path = "/Users/kev29.06/Documents/Cuatrimestre-8/Inteligencia Artificial/ia-perceptron/dataset/dataset.csv"
    x, y_o = load_data(path)

    K = 500
    lambdas = [1e-6, 0.25, 0.5, 0.75, 1]

    all_errors = []
    weights_summary = []

    for lr in lambdas:
        errors, weights_history = train_with_lambda(x, y_o, lr, K)

        # Guardar pesos iniciales y finales
        w_initial = weights_history[0]
        w_final = weights_history[-1]
        weights_summary.append({
            'lambda': lr,
            **{f'w{i}_init': w_initial[i] for i in range(len(w_initial))},
            **{f'w{i}_final': w_final[i] for i in range(len(w_final))}
        })

        if lr == 1e-6:
            # Graficar pesos
            plot_weights(weights_history, lr)

        # Guardar errores para gráfica global
        all_errors.append((lr, errors))

    # Graficar errores global
    plot_errors(all_errors, K)

    # Guardar dataset de pesos iniciales y finales
    save_weights_summary(weights_summary)

if __name__ == "__main__":
    main()


