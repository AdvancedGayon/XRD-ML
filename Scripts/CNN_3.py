import os
import glob
import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Configuración del estilo académico para todos los gráficos
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'Times New Roman', 'Palatino'],
    'font.size': 11,
    'mathtext.fontset': 'cm',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.linewidth': 1.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'gray',
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'figure.figsize': (8, 6),
    'figure.autolayout': True,
})

# Paleta de colores académica
COLOR_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#F0E442']

# 1. Modelo de CNN optimizado
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # En lugar de calcular automáticamente, usaremos una capa de adaptación
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)  # Forzamos una salida de tamaño fijo

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 100, 256),  # Ahora siempre será 64 * 100
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.adaptive_pool(x)  # Asegura que la salida tenga el tamaño esperado
        x = self.fc(x)
        return x

class CsLoss(nn.Module):
    def __init__(self, alpha=1000.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, outputs, labels):
        mse_loss = self.mse(outputs, labels)
        cs_loss = self.alpha * (1 - torch.nn.functional.cosine_similarity(outputs, labels).mean()) / 2
        cs_mse_loss = mse_loss + cs_loss
        return cs_mse_loss

# 2. Carga de Datos mejorada
def load_data(directory):
    """
    Carga los datos desde un directorio y los convierte a tensores de PyTorch.
    """
    files = glob.glob(os.path.join(directory, '*.json'))
    XRDs, Quants = [], []

    print(f"Cargando {len(files)} archivos de {directory}...")

    for file in files:
        try:
            with open(file, 'r') as f:
                patt = json.load(f)
            XRDs.append(patt['xrd'])
            Quants.append(patt['q'])
        except Exception as e:
            print(f"Error al cargar {file}: {e}")

    # Convertir listas en arrays de NumPy antes de pasarlos a tensores
    X = torch.tensor(np.array(XRDs), dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(np.array(Quants), dtype=torch.float32)

    print(f"Datos cargados: X shape = {X.shape}, y shape = {y.shape}")

    return TensorDataset(X, y)

# 3. Función de entrenamiento
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    """
    Entrena el modelo y devuelve las pérdidas de entrenamiento y validación.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        # Modo entrenamiento
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Modo evaluación
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Actualizar scheduler
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        print(f"Época {epoch+1}/{epochs}, Pérdida Train: {train_loss:.4f}, Pérdida Val: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Cargar el mejor modelo
    model.load_state_dict(best_model_state)

    return model, train_losses, val_losses

# 4. Función de evaluación del modelo
def evaluate_model(model, data_loader, device):
    """
    Evalúa el modelo y devuelve las métricas de rendimiento.
    """
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return y_true, y_pred, rmse, mae, r2

# 5. Funciones para graficar con estilo académico
def plot_training_history(train_losses, val_losses, save_path):
    """
    Genera un gráfico de calidad académica del historial de entrenamiento.
    """
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Graficar pérdidas
    ax.plot(epochs, train_losses, 'o-', color=COLOR_PALETTE[0], linewidth=2,
            markersize=6, label='Pérdida de entrenamiento')
    ax.plot(epochs, val_losses, 's-', color=COLOR_PALETTE[1], linewidth=2,
            markersize=6, label='Pérdida de validación')

    # Mejorar etiquetas y título
    ax.set_xlabel('Época', fontweight='medium')
    ax.set_ylabel('Pérdida (MSE + Coseno)', fontweight='medium')
    ax.set_title('Evolución de la pérdida durante el entrenamiento', fontweight='bold', pad=15)

    # Optimizar ejes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')

    # Añadir leyenda
    leg = ax.legend(loc='upper right', frameon=True, fancybox=False,
                   framealpha=0.95, edgecolor='gray')

    # Información adicional
    min_val_loss = min(val_losses)
    min_val_epoch = val_losses.index(min_val_loss) + 1
    ax.annotate(f'Mejor pérdida: {min_val_loss:.4f}',
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch+1, min_val_loss*1.2),
                arrowprops=dict(arrowstyle='->'),
                fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return fig

def plot_predictions(y_true, y_pred, save_path):
    """
    Genera un gráfico de dispersión de predicciones vs valores reales.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Calcular métricas
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Graficar los puntos de datos
    ax.scatter(y_true, y_pred, s=60, alpha=0.6, color=COLOR_PALETTE[0],
               edgecolor='black', linewidth=0.5, label="Datos")

    # Línea ideal (y = x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label="Ideal (y = x)")

    # Mejorar etiquetas y título
    ax.set_xlabel('Valores Reales', fontweight='medium')
    ax.set_ylabel('Predicciones', fontweight='medium')
    ax.set_title('Predicciones vs Valores Reales', fontweight='bold', pad=15)

    # Optimizar ejes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.set_aspect('equal')

    # Añadir métricas como texto
    metrics_text = f"RMSE = {rmse:.4f}\nMAE = {mae:.4f}\n$R^2$ = {r2:.4f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return fig

def plot_component_predictions(y_true, y_pred, save_path, component_names=None):
    """
    Genera un gráfico de dispersión de predicciones vs valores reales para cada componente.
    """
    num_components = y_true.shape[1]
    if component_names is None:
        component_names = [f"Componente {i+1}" for i in range(num_components)]

    fig, axes = plt.subplots(2, (num_components + 1) // 2, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(num_components):
        ax = axes[i]
        y_true_comp = y_true[:, i]
        y_pred_comp = y_pred[:, i]

        # Calcular métricas por componente
        rmse = np.sqrt(mean_squared_error(y_true_comp, y_pred_comp))
        r2 = r2_score(y_true_comp, y_pred_comp)

        # Graficar los puntos de datos
        ax.scatter(y_true_comp, y_pred_comp, s=40, alpha=0.6, color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                   edgecolor='black', linewidth=0.5)

        # Línea ideal (y = x)
        min_val = min(np.min(y_true_comp), np.min(y_pred_comp))
        max_val = max(np.max(y_true_comp), np.max(y_pred_comp))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5)

        # Mejorar etiquetas y título
        ax.set_xlabel('Valor Real', fontsize=10)
        ax.set_ylabel('Predicción', fontsize=10)
        ax.set_title(f'{component_names[i]}', fontweight='medium')

        # Optimizar ejes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')

        # Añadir métricas como texto
        metrics_text = f"RMSE = {rmse:.4f}\n$R^2$ = {r2:.4f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

    # Ocultar ejes no utilizados
    for i in range(num_components, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return fig


# CONFIGURACIÓN PRINCIPAL
def main():
    # Directorios y rutas de modelos (pueden ser modificados según necesidad)
    EXP_DIR = r"D:\Desktop\U\Tesis\Datos\XRD\Experimental\json\Train"
    SYNT_DIR = r"D:\Desktop\U\Tesis\Datos\XRD\Synthetic"
    RESULTS_DIR = r"D:\Desktop\U\Tesis\Resultados\ML\3_2"

    # Crear directorios para los resultados
    os.makedirs(os.path.join(RESULTS_DIR, "all_data"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "only_exp"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "only_synt"), exist_ok=True)

    # Rutas para modelos
    ALL_DATA_MODEL = os.path.join(RESULTS_DIR, "all_data", "all_data_model.pth")
    EXP_DATA_MODEL = os.path.join(RESULTS_DIR, "only_exp", "exp_data_model.pth")
    SYNT_DATA_MODEL = os.path.join(RESULTS_DIR, "only_synt", "synt_data_model.pth")

    # Parámetros del modelo
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {DEVICE}")

    # Hiperparámetros optimizados
    config = {
        "train_size": 33709, #184 datos experimentales, el restante es sintético
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 0.001,  # Reducido para mayor estabilidad
        "weight_decay": 1e-5,    # Añadido regularización L2
        "num_classes": 5,        # Número de clases (output dimension)
        "validation_split": 0.2, # Proporción para validación
        "random_seed": 42,       # Semilla para reproducibilidad
    }

    # Fijar semillas para reproducibilidad
    random.seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    torch.manual_seed(config["random_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["random_seed"])

    # Cargar datos
    print("Cargando datasets...")
    try:
        exp_dataset = load_data(EXP_DIR)
        synt_dataset = load_data(SYNT_DIR)

        # Crear dataset combinado
        all_dataset = torch.utils.data.ConcatDataset([exp_dataset, synt_dataset])

        # Implementar función para entrenar diferentes modelos
        train_evaluate_model("all_data", all_dataset, ALL_DATA_MODEL, config, DEVICE, RESULTS_DIR)
        train_evaluate_model("only_exp", exp_dataset, EXP_DATA_MODEL, config, DEVICE, RESULTS_DIR)
        train_evaluate_model("only_synt", synt_dataset, SYNT_DATA_MODEL, config, DEVICE, RESULTS_DIR)

    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

def train_evaluate_model(name, dataset, model_path, config, device, results_dir):
    """
    Entrena y evalúa un modelo con un dataset específico.
    """
    print(f"\n{'='*50}\nEntrenando modelo con dataset: {name}\n{'='*50}")

    # Dividir dataset en entrenamiento y validación
    train_size = min(config["train_size"], len(dataset))
    indices = list(range(len(dataset)))

    # Si el dataset es más grande que train_size, seleccionamos una muestra aleatoria
    if len(dataset) > train_size:
        indices = random.sample(indices, train_size)

    # Dividir en entrenamiento y validación
    train_indices, val_indices = train_test_split(
        indices, test_size=config["validation_split"], random_state=config["random_seed"]
    )

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config["batch_size"], shuffle=False)

    print(f"Tamaño de dataset de entrenamiento: {len(train_subset)}")
    print(f"Tamaño de dataset de validación: {len(val_subset)}")

    # Inicializar modelo
    model = CNN(in_channels=1, num_classes=config["num_classes"]).to(device)

    # Inicializar optimizador y pérdida
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    criterion = CsLoss(alpha=1000.0)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # Entrenar modelo
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, config["epochs"]
    )

    # Guardar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses
    }, model_path)
    print(f"Modelo guardado en {model_path}")

    # Evaluar modelo
    y_true, y_pred, rmse, mae, r2 = evaluate_model(model, val_loader, device)

    print(f"Resultados de evaluación:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    # Generar gráficos con estilo académico
    results_subdir = os.path.join(results_dir, name)
    os.makedirs(results_subdir, exist_ok=True)

    # Gráfico de pérdidas
    plot_training_history(
        train_losses, val_losses,
        os.path.join(results_subdir, f"{name}_training_loss.pdf")
    )

    # Gráfico de predicciones vs valores reales
    plot_predictions(
        y_true, y_pred,
        os.path.join(results_subdir, f"{name}_pred_vs_actual.pdf")
    )

    # Gráfico por componentes
    component_names = ["Cuarzo", "Calcita", "Pirita", "Magnetita", "Otro"]
    plot_component_predictions(
        y_true, y_pred,
        os.path.join(results_subdir, f"{name}_component_predictions.pdf"),
        component_names=component_names
    )

    print(f"Gráficos generados y guardados en {results_subdir}")

if __name__ == "__main__":
    main()
