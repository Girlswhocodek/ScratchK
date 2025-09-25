"""
IMPLEMENTACIÓN DE REGRESIÓN LINEAL DESDE CERO
Problemas 1 al 11 - Scratch Linear Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# =============================================================================
# [PROBLEMA 4] Función MSE
# =============================================================================
def MSE(y_pred, y):
    """
    Error Cuadrático Medio (Mean Squared Error)
    """
    return np.mean((y_pred - y) ** 2)

# =============================================================================
# [PROBLEMAS 1-3, 5] Clase ScratchLinearRegression
# =============================================================================
class ScratchLinearRegression():
    """
    Implementación desde cero de Regresión Lineal
    """
    
    def __init__(self, num_iter=1000, lr=0.01, no_bias=False, verbose=False):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None
    
    # [PROBLEMA 1] Función de suposición
    def _linear_hypothesis(self, X):
        """Calcula la hipótesis lineal hθ(x) = θᵀx"""
        return X @ self.coef_
    
    # [PROBLEMA 2] Método de descenso de gradiente
    def _gradient_descent(self, X, error):
        """Actualiza parámetros usando descenso de gradiente"""
        m = X.shape[0]
        gradient = (X.T @ error) / m
        self.coef_ = self.coef_ - self.lr * gradient
    
    # [PROBLEMA 5] Función objetivo
    def _objective_function(self, y, y_pred):
        """Función de pérdida J(θ) = 1/(2m) * Σ(hθ(xⁱ) - yⁱ)²"""
        m = len(y)
        return np.sum((y_pred - y) ** 2) / (2 * m)
    
    def fit(self, X, y, X_val=None, y_val=None):
        """Entrena el modelo de regresión lineal"""
        # [PROBLEMA 8] Manejo del bias term
        if not self.no_bias:
            X = np.c_[np.ones(X.shape[0]), X]
            if X_val is not None:
                X_val = np.c_[np.ones(X_val.shape[0]), X_val]
        
        # Inicializar parámetros
        n_features = X.shape[1]
        self.coef_ = np.random.randn(n_features)
        
        # Loop de entrenamiento
        for i in range(self.iter):
            # [PROBLEMA 1] Calcular predicciones
            y_pred = self._linear_hypothesis(X)
            
            # [PROBLEMA 5] Calcular pérdida
            self.loss[i] = self._objective_function(y, y_pred)
            
            # Pérdida de validación
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                self.val_loss[i] = self._objective_function(y_val, y_val_pred)
            
            # [PROBLEMA 2] Actualizar parámetros
            error = y_pred - y
            self._gradient_descent(X, error)
            
            # Verbose output
            if self.verbose and i % 100 == 0:
                if X_val is not None and y_val is not None:
                    print(f"Iter {i}: Train Loss = {self.loss[i]:.6f}, Val Loss = {self.val_loss[i]:.6f}")
                else:
                    print(f"Iter {i}: Train Loss = {self.loss[i]:.6f}")
    
    # [PROBLEMA 3] Función de predicción
    def predict(self, X):
        """Realiza predicciones con el modelo entrenado"""
        if not self.no_bias and X.shape[1] == self.coef_.shape[0] - 1:
            X = np.c_[np.ones(X.shape[0]), X]
        return self._linear_hypothesis(X)

# =============================================================================
# [PROBLEMA 7] Función para curvas de aprendizaje
# =============================================================================
def plot_learning_curve(train_loss, val_loss=None, title="Curva de Aprendizaje", save_path=None):
    """Grafica la curva de aprendizaje"""
    plt.figure(figsize=(10, 6))
    iterations = range(len(train_loss))
    
    plt.plot(iterations, train_loss, label='Pérdida Entrenamiento', linewidth=2)
    
    if val_loss is not None:
        plt.plot(iterations, val_loss, label='Pérdida Validación', linewidth=2)
    
    plt.xlabel('Iteraciones')
    plt.ylabel('Pérdida')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado: {save_path}")
    plt.close()

# =============================================================================
# [PROBLEMA 9] Función para características polinómicas
# =============================================================================
def create_polynomial_features(X, degree=2):
    """Crea características polinómicas"""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    scaler = StandardScaler()
    return scaler.fit_transform(X_poly)

# =============================================================================
# [PROBLEMA 10] Derivación de la fórmula de actualización
# =============================================================================
def derivacion_formula_actualizacion():
    """
    Deriva matemáticamente la fórmula de actualización del descenso de gradiente
    """
    print("\n" + "="*70)
    print("PROBLEMA 10: DERIVACIÓN MATEMÁTICA DE LA FÓRMULA DE ACTUALIZACIÓN")
    print("="*70)
    
    # Explicación paso a paso con notación matemática correcta
    derivacion = """
    OBJETIVO: Derivar la fórmula de actualización
    
        θ_j := θ_j - α * (1/m) * Σ[hθ(xⁱ) - yⁱ] * x_jⁱ
    
    donde:
        θ_j: parámetro j-ésimo
        α: tasa de aprendizaje (learning rate)
        m: número de ejemplos de entrenamiento
        hθ(xⁱ): predicción para el ejemplo i
        yⁱ: valor real para el ejemplo i
        x_jⁱ: característica j del ejemplo i

    PASO 1: FÓRMULA GENERAL DEL DESCENSO DE GRADIENTE
    
        θ_j := θ_j - α * ∂J(θ)/∂θ_j

    PASO 2: FUNCIÓN DE PÉRDIDA J(θ)

        J(θ) = (1/(2m)) * Σ[hθ(xⁱ) - yⁱ]²

    PASO 3: CALCULAR LA DERIVADA PARCIAL ∂J(θ)/∂θ_j

        ∂J(θ)/∂θ_j = ∂/∂θ_j [ (1/(2m)) * Σ[hθ(xⁱ) - yⁱ]² ]

    PASO 4: APLICAR REGLA DE LA CADENA

        = (1/(2m)) * Σ 2[hθ(xⁱ) - yⁱ] * ∂/∂θ_j [hθ(xⁱ) - yⁱ]

        = (1/m) * Σ [hθ(xⁱ) - yⁱ] * ∂/∂θ_j hθ(xⁱ)

    PASO 5: DERIVAR LA FUNCIÓN DE HIPÓTESIS hθ(xⁱ)

        hθ(xⁱ) = θ₀ + θ₁x₁ⁱ + θ₂x₂ⁱ + ... + θ_jx_jⁱ + ... + θ_nx_nⁱ

        ∂/∂θ_j hθ(xⁱ) = ∂/∂θ_j [θ₀ + θ₁x₁ⁱ + ... + θ_jx_jⁱ + ... + θ_nx_nⁱ]
                       = x_jⁱ

    PASO 6: SUSTITUIR EN LA DERIVADA

        ∂J(θ)/∂θ_j = (1/m) * Σ [hθ(xⁱ) - yⁱ] * x_jⁱ

    PASO 7: SUSTITUIR EN LA FÓRMULA DEL DESCENSO DE GRADIENTE

        θ_j := θ_j - α * ∂J(θ)/∂θ_j
              = θ_j - α * (1/m) * Σ [hθ(xⁱ) - yⁱ] * x_jⁱ

    ¡FÓRMULA DERIVADA!

    DEMOSTRACIÓN EN NOTACIÓN VECTORIAL/MATRICIAL:

    Para todos los parámetros simultáneamente:

        ∇J(θ) = (1/m) * Xᵀ(Xθ - y)

        θ := θ - α * ∇J(θ)
            = θ - α * (1/m) * Xᵀ(Xθ - y)

    donde:
        X: matriz de diseño (m × n+1)
        y: vector de valores reales (m × 1)
        θ: vector de parámetros ((n+1) × 1)

    VERIFICACIÓN DE LA IMPLEMENTACIÓN:

    En nuestro código, en el método _gradient_descent:

        error = hθ(X) - y = Xθ - y
        gradient = (X.T @ error) / m = (1/m) * Xᵀ(Xθ - y)
        self.coef_ = self.coef_ - self.lr * gradient

    ¡Que es exactamente la fórmula derivada!
    """
    print(derivacion)
    
    # Demostración con ejemplo numérico simple
    print("\n" + "-"*50)
    print("DEMOSTRACIÓN NUMÉRICA CON EJEMPLO SIMPLE")
    print("-"*50)
    
    # Crear ejemplo mínimo
    np.random.seed(42)
    m = 5  # 5 ejemplos
    n = 2  # 2 características (más bias)
    
    # Datos de ejemplo
    X = np.c_[np.ones(m), np.random.randn(m, n)]  # X con bias term
    y = np.random.randn(m)
    theta = np.random.randn(n+1)
    alpha = 0.01
    
    print(f"Matriz X (con bias):\n{X}")
    print(f"Vector y: {y}")
    print(f"Parámetros θ iniciales: {theta}")
    print(f"Tasa de aprendizaje α: {alpha}")
    
    # Calcular manualmente la derivada
    predictions = X @ theta
    error = predictions - y
    gradient_manual = (X.T @ error) / m
    
    print(f"\nCálculo manual del gradiente:")
    print(f"Predicciones hθ(X): {predictions}")
    print(f"Error (hθ(X) - y): {error}")
    print(f"Gradiente ∇J(θ) = (1/m) * Xᵀ(Xθ - y): {gradient_manual}")
    
    # Mostrar la actualización
    theta_updated = theta - alpha * gradient_manual
    print(f"\nActualización:")
    print(f"θ_nuevo = θ - α * ∇J(θ) = {theta_updated}")
    
    # Verificar con nuestra implementación
    model = ScratchLinearRegression(num_iter=1, lr=alpha, verbose=False)
    model.coef_ = theta.copy()
    model._gradient_descent(X, error)
    
    print(f"\nVerificación con nuestra implementación:")
    print(f"θ después de _gradient_descent: {model.coef_}")
    print(f"¿Coinciden? {np.allclose(theta_updated, model.coef_)}")
    
    # Explicación adicional
    print("\n" + "-"*50)
    print("INTERPRETACIÓN INTUITIVA")
    print("-"*50)
    interpretacion = """
    SIGNIFICADO INTUITIVO DE LA FÓRMULA:
    
    La actualización para cada parámetro θ_j es:
    
        θ_j := θ_j - α * (1/m) * Σ errorⁱ * x_jⁱ
    
    Donde:
        - errorⁱ = hθ(xⁱ) - yⁱ: qué tan equivocada está nuestra predicción
        - x_jⁱ: valor de la característica j en el ejemplo i
        - (1/m): promedio sobre todos los ejemplos
        - α: tamaño del paso (qué tan rápido aprendemos)
    
    INTERPRETACIÓN:
    - Si el error es grande y positivo (sobrestimamos), reducimos θ_j
    - Si el error es grande y negativo (subestimamos), aumentamos θ_j  
    - El ajuste es proporcional al valor de la característica x_jⁱ
    - Aprendemos del promedio de todos los ejemplos (no solo uno)
    
    Esto hace que el algoritmo aprenda de manera estable y converja al óptimo.
    """
    print(interpretacion)

# =============================================================================
# [PROBLEMA 11] Explicación de convexidad
# =============================================================================
def demostrar_convexidad():
    """
    Demuestra por qué la regresión lineal no tiene óptimos locales
    mediante fórmulas y gráficos
    """
    print("\n" + "="*70)
    print("PROBLEMA 11: CONVEXIDAD Y FALTA DE ÓPTIMOS LOCALES")
    print("="*70)
    
    # Explicación teórica
    explicacion = """
    ¿POR QUÉ LA REGRESIÓN LINEAL NO TIENE ÓPTIMOS LOCALES?
    
    RAZÓN PRINCIPAL: La función de pérdida J(θ) es CONVEXA
    
    1. FUNCIÓN DE PÉRDIDA:
       J(θ) = (1/(2m)) * Σ(hθ(xⁱ) - yⁱ)²
            = (1/(2m)) * (Xθ - y)ᵀ(Xθ - y)
    
    2. MATRIZ HESSIANA (segundas derivadas):
       ∇²J(θ) = (1/m) * XᵀX
    
    3. PROPIEDAD CLAVE: XᵀX es SEMIDEFINIDA POSITIVA
       - Todos sus valores propios son ≥ 0
       - Por lo tanto, ∇²J(θ) es semidefinida positiva
       - Esto implica que J(θ) es CONVEXA
    
    4. CONSECUENCIAS DE LA CONVEXIDAD:
       - Solo existe UN MÍNIMO GLOBAL
       - No hay mínimos locales
       - El descenso de gradiente SIEMPRE converge al óptimo
       - Independientemente del punto inicial
    
    COMPARACIÓN CON MODELOS NO CONVEXOS (ej. redes neuronales):
       - Pueden tener múltiples mínimos locales
       - El gradiente puede quedar atrapado
       - La convergencia al óptimo global no está garantizada
    """
    print(explicacion)
    
    # Demostración visual con gráficos
    print("\nGenerando demostración visual de la convexidad...")
    
    # Crear datos de ejemplo simples (2 parámetros)
    np.random.seed(42)
    X_simple = 2 * np.random.rand(50, 1)
    y_simple = 3 + 4 * X_simple + np.random.randn(50, 1) * 0.5
    
    # Añadir bias term
    X_with_bias = np.c_[np.ones((50, 1)), X_simple]
    
    # Crear grid de parámetros para visualización
    theta0_vals = np.linspace(0, 6, 50)  # θ₀ (bias)
    theta1_vals = np.linspace(0, 8, 50)  # θ₁ (pendiente)
    Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)
    
    # Calcular función de pérdida para cada combinación de parámetros
    J_vals = np.zeros_like(Theta0)
    
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            theta = np.array([Theta0[i, j], Theta1[i, j]])
            predictions = X_with_bias @ theta
            J_vals[i, j] = np.mean((predictions - y_simple.flatten())**2) / 2
    
    # Gráfico 1: Superficie 3D de la función de pérdida
    fig = plt.figure(figsize=(15, 5))
    
    # Subplot 1: Superficie 3D
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(Theta0, Theta1, J_vals, cmap='viridis', 
                           alpha=0.8, linewidth=0, antialiased=True)
    ax1.set_xlabel('θ₀ (Bias)')
    ax1.set_ylabel('θ₁ (Pendiente)')
    ax1.set_zlabel('J(θ)')
    ax1.set_title('Superficie Convexa de la\nFunción de Pérdida')
    
    # Marcar el mínimo global (solución analítica)
    theta_optimal = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y_simple
    min_J = np.mean((X_with_bias @ theta_optimal - y_simple.flatten())**2) / 2
    ax1.scatter(theta_optimal[0], theta_optimal[1], min_J, 
               color='red', s=100, label='Mínimo Global')
    
    # Subplot 2: Curvas de nivel
    ax2 = fig.add_subplot(132)
    contour = ax2.contour(Theta0, Theta1, J_vals, levels=20)
    ax2.clabel(contour, inline=1, fontsize=8)
    ax2.set_xlabel('θ₀ (Bias)')
    ax2.set_ylabel('θ₁ (Pendiente)')
    ax2.set_title('Curvas de Nivel - Forma de "Bowl"\n(Indica Convexidad)')
    ax2.grid(True, alpha=0.3)
    
    # Marcar el mínimo en las curvas de nivel
    ax2.scatter(theta_optimal[0], theta_optimal[1], color='red', s=100, 
               label='Mínimo Global')
    ax2.legend()
    
    # Subplot 3: Demostración de múltiples puntos iniciales
    ax3 = fig.add_subplot(133)
    
    # Puntos iniciales diferentes
    initial_points = [
        np.array([1.0, 2.0]),   # Punto 1
        np.array([5.0, 1.0]),   # Punto 2  
        np.array([2.0, 7.0]),   # Punto 3
        np.array([4.5, 6.0])    # Punto 4
    ]
    
    colors = ['blue', 'green', 'orange', 'purple']
    
    # Mostrar curvas de nivel
    contour = ax3.contour(Theta0, Theta1, J_vals, levels=15)
    
    for i, (point, color) in enumerate(zip(initial_points, colors)):
        # Simular trayectoria de descenso de gradiente (simplificado)
        current_point = point.copy()
        trajectory = [current_point.copy()]
        
        for step in range(10):  # 10 pasos de gradiente
            # Calcular gradiente (simplificado)
            theta_current = current_point
            predictions = X_with_bias @ theta_current
            error = predictions - y_simple.flatten()
            gradient = (X_with_bias.T @ error) / len(y_simple)
            
            # Actualizar
            current_point = current_point - 0.1 * gradient
            trajectory.append(current_point.copy())
        
        trajectory = np.array(trajectory)
        ax3.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color=color, 
                linewidth=2, markersize=4, label=f'Punto inicial {i+1}')
        ax3.scatter(trajectory[0, 0], trajectory[0, 1], color=color, s=100, 
                   marker='s', edgecolors='black')
    
    ax3.scatter(theta_optimal[0], theta_optimal[1], color='red', s=150, 
               marker='*', label='Mínimo Global', edgecolors='black')
    ax3.set_xlabel('θ₀ (Bias)')
    ax3.set_ylabel('θ₁ (Pendiente)')
    ax3.set_title('Múltiples Trayectorias de\nDescenso de Gradiente')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graficos/demostracion_convexidad.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico guardado: graficos/demostracion_convexidad.png")
    
    # Gráfico 4: Comparación con función no convexa
    fig = plt.figure(figsize=(12, 5))
    
    # Función convexa (nuestra pérdida)
    ax1 = fig.add_subplot(121)
    contour1 = ax1.contour(Theta0, Theta1, J_vals, levels=15)
    ax1.set_xlabel('θ₀')
    ax1.set_ylabel('θ₁')
    ax1.set_title('FUNCIÓN CONVEXA\n(Regresión Lineal)')
    ax1.grid(True, alpha=0.3)
    
    # Función no convexa de ejemplo (Rastrigin)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y)) + 20
    
    ax2 = fig.add_subplot(122)
    contour2 = ax2.contour(X, Y, Z, levels=15)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('FUNCIÓN NO CONVEXA\n(Múltiples mínimos locales)')
    ax2.grid(True, alpha=0.3)
    
    # Marcar algunos mínimos locales
    local_minima = [(-3.5, -3.5), (0, 0), (3.5, 3.5)]
    for min_point in local_minima:
        ax2.scatter(min_point[0], min_point[1], color='red', s=50)
    
    plt.tight_layout()
    plt.savefig('graficos/comparacion_convexidad.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Gráfico guardado: graficos/comparacion_convexidad.png")
    
    # Explicación adicional
    print("\n" + "-"*50)
    print("DEMOSTRACIÓN COMPLETADA")
    print("-"*50)
    print("Los gráficos muestran:")
    print("1. La superficie CONVEXA de la función de pérdida")
    print("2. Cómo MÚLTIPLES puntos iniciales convergen al MISMO mínimo")
    print("3. Comparación con funciones NO CONVEXAS (múltiples mínimos)")
    print("4. Esto explica por qué el aprendizaje continuo SIEMPRE encuentra el óptimo")


# =============================================================================
# Función para gráficos de comparación de predicciones
# =============================================================================
def crear_graficos_comparacion(y_test, y_pred_scratch, y_pred_sklearn, save_dir="graficos"):
    """Crea todos los gráficos de comparación de predicciones"""
    
    # Calcular diferencias
    differences = np.abs(y_pred_scratch - y_pred_sklearn)
    
    # 1. Gráfico: Predicciones vs Valores Reales
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Prediction vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred_scratch, alpha=0.5, label='Scratch', color='blue', s=20)
    plt.scatter(y_test, y_pred_sklearn, alpha=0.5, label='Sklearn', color='red', s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', alpha=0.8, linewidth=2)
    plt.xlabel('Valor Real')
    plt.ylabel('Predicción')
    plt.title('Predicciones vs Valores Reales\n(Scratch vs Sklearn)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Distribution of Prediction Differences
    plt.subplot(1, 3, 2)
    plt.hist(differences, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Diferencia entre Predicciones')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Diferencias\nentre Scratch y Sklearn')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Comparación directa de predicciones
    plt.subplot(1, 3, 3)
    plt.scatter(y_pred_scratch, y_pred_sklearn, alpha=0.5, color='purple')
    plt.plot([y_pred_scratch.min(), y_pred_scratch.max()], 
             [y_pred_scratch.min(), y_pred_scratch.max()], 'k--', alpha=0.8)
    plt.xlabel('Predicciones Scratch')
    plt.ylabel('Predicciones Sklearn')
    plt.title('Scratch vs Sklearn\nComparación Directa')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparacion_predicciones.png")
    plt.close()
    print(f"Gráfico guardado: {save_dir}/comparacion_predicciones.png")
    
    # 2. Gráfico: Errores de predicción
    plt.figure(figsize=(12, 5))
    
    errors_scratch = np.abs(y_test - y_pred_scratch)
    errors_sklearn = np.abs(y_test - y_pred_sklearn)
    
    plt.subplot(1, 2, 1)
    plt.hist(errors_scratch, bins=30, alpha=0.7, label='Scratch', color='blue')
    plt.hist(errors_sklearn, bins=30, alpha=0.7, label='Sklearn', color='red')
    plt.xlabel('Error Absoluto')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Errores Absolutos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.boxplot([errors_scratch, errors_sklearn], labels=['Scratch', 'Sklearn'])
    plt.ylabel('Error Absoluto')
    plt.title('Comparación de Errores (Boxplot)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/distribucion_errores.png")
    plt.close()
    print(f"Gráfico guardado: {save_dir}/distribucion_errores.png")

# =============================================================================
# Datos de prueba
# =============================================================================
def cargar_datos_habitacionales():
    """Crea datos de ejemplo para precios de viviendas"""
    np.random.seed(42)
    n_samples = 1000
    
    X = np.column_stack([
        np.random.normal(150, 50, n_samples),  # Tamaño (m²)
        np.random.normal(3, 1, n_samples),     # Habitaciones
        np.random.normal(2, 0.5, n_samples),   # Baños
        np.random.normal(20, 5, n_samples),    # Antigüedad
        np.random.normal(10, 3, n_samples)     # Distancia al centro
    ])
    
    true_coef = np.array([1000, 50000, 30000, -10000, -20000])
    y = X @ true_coef + np.random.normal(0, 50000, n_samples)
    
    return X, y

# =============================================================================
# FUNCIÓN PRINCIPAL - EJECUTA TODOS LOS PROBLEMAS
# =============================================================================
def main():
    """Función principal que ejecuta todos los problemas del 1 al 11"""
    print("REGERSIÓN LINEAL DESDE CERO - PROBLEMAS 1 AL 11")
    print("="*70)
    
    # Crear carpeta para gráficos
    os.makedirs('graficos', exist_ok=True)
    
    # Cargar datos
    print("\n1. Cargando datos de precios de viviendas...")
    X, y = cargar_datos_habitacionales()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Estandarizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Datos de entrenamiento: {X_train_scaled.shape}")
    print(f"   Datos de prueba: {X_test_scaled.shape}")
    
    # [PROBLEMA 6] Entrenar y comparar modelos
    print("\n2. Entrenando modelo Scratch...")
    scratch_model = ScratchLinearRegression(num_iter=1000, lr=0.01, verbose=True)
    scratch_model.fit(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Predicciones
    y_pred_scratch = scratch_model.predict(X_test_scaled)
    mse_scratch = MSE(y_pred_scratch, y_test)
    
    # Comparación con Scikit-learn
    print("\n3. Entrenando modelo Scikit-learn...")
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_scaled, y_train)
    y_pred_sklearn = sklearn_model.predict(X_test_scaled)
    mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
    
    print(f"\n4. Comparación de resultados:")
    print(f"   MSE Scratch: {mse_scratch:.6f}")
    print(f"   MSE Sklearn: {mse_sklearn:.6f}")
    print(f"   Diferencia: {abs(mse_scratch - mse_sklearn):.6f}")
    
    # [PROBLEMA 7] Curva de aprendizaje
    print("\n5. Generando curva de aprendizaje...")
    plot_learning_curve(scratch_model.loss, scratch_model.val_loss,
                       "Curva de Aprendizaje - Regresión Lineal",
                       "graficos/curva_aprendizaje.png")
    
    # [PROBLEMA 6] Gráficos de comparación de predicciones
    print("\n6. Generando gráficos de comparación...")
    crear_graficos_comparacion(y_test, y_pred_scratch, y_pred_sklearn)
    
    # [PROBLEMA 8] Análisis del bias term
    print("\n7. Analizando efecto del bias term...")
    
    # Con bias term
    model_with_bias = ScratchLinearRegression(num_iter=500, lr=0.01, no_bias=False, verbose=False)
    model_with_bias.fit(X_train_scaled, y_train, X_test_scaled, y_test)
    y_pred_with_bias = model_with_bias.predict(X_test_scaled)
    mse_with_bias = MSE(y_pred_with_bias, y_test)
    
    # Sin bias term
    model_no_bias = ScratchLinearRegression(num_iter=500, lr=0.01, no_bias=True, verbose=False)
    model_no_bias.fit(X_train_scaled, y_train, X_test_scaled, y_test)
    y_pred_no_bias = model_no_bias.predict(X_test_scaled)
    mse_no_bias = MSE(y_pred_no_bias, y_test)
    
    print(f"   MSE con bias: {mse_with_bias:.6f}")
    print(f"   MSE sin bias: {mse_no_bias:.6f}")
    print(f"   Mejora: {((mse_no_bias - mse_with_bias)/mse_no_bias*100):.2f}%")
    
    # Gráfico bias term
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model_with_bias.loss, label='Con bias')
    plt.title('Con Bias Term')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(model_no_bias.loss, label='Sin bias')
    plt.title('Sin Bias Term')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('graficos/comparacion_bias.png')
    plt.close()
    print("   Gráfico guardado: graficos/comparacion_bias.png")
    
    # [PROBLEMA 9] Características polinómicas
    print("\n8. Probando características polinómicas...")
    degrees = [1, 2]
    
    for degree in degrees:
        try:
            X_train_poly = create_polynomial_features(X_train_scaled, degree)
            X_test_poly = create_polynomial_features(X_test_scaled, degree)
            
            model_poly = ScratchLinearRegression(num_iter=800, lr=0.001, verbose=False)
            model_poly.fit(X_train_poly, y_train, X_test_poly, y_test)
            
            y_pred_poly = model_poly.predict(X_test_poly)
            mse_poly = MSE(y_pred_poly, y_test)
            
            print(f"   Grado {degree}: MSE = {mse_poly:.6f}")
            
        except Exception as e:
            print(f"   Error con grado {degree}: {e}")
    
    # [PROBLEMA 10] Explicación derivación
    derivacion_formula_actualizacion()
    
    # [PROBLEMA 11] Explicación convexidad
    demostrar_convexidad()
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN EJECUCIÓN - PROBLEMAS 1 AL 11")
    print("="*70)
    print("✅ Problema 1: Función de suposición")
    print("✅ Problema 2: Descenso de gradiente") 
    print("✅ Problema 3: Función de predicción")
    print("✅ Problema 4: Función MSE")
    print("✅ Problema 5: Función objetivo")
    print("✅ Problema 6: Entrenamiento y validación")
    print("✅ Problema 7: Curvas de aprendizaje")
    print("✅ Problema 8: Análisis de bias term")
    print("✅ Problema 9: Características polinómicas")
    print("✅ Problema 10: Derivación matemática")
    print("✅ Problema 11: Convexidad con demostración gráfica")
    print("="*70)
    print("\nGRÁFICOS GENERADOS:")
    print("📊 graficos/curva_aprendizaje.png")
    print("📊 graficos/comparacion_predicciones.png")
    print("📊 graficos/distribucion_errores.png") 
    print("📊 graficos/comparacion_bias.png")
    print("📊 graficos/demostracion_convexidad.png")  # NUEVO
    print("📊 graficos/comparacion_convexidad.png") 
    print("\nEjecución completada exitosamente! ")

if __name__ == "__main__":
    main()