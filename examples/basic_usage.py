#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemplo Básico de Uso da GimnTools
Demonstra funcionalidades fundamentais da biblioteca
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Importações da GimnTools
try:
    from GimnTools.ImaGIMN.image import image
    from GimnTools.ImaGIMN.processing.filters.spatial_filters import apply_filter
    from GimnTools.ImaGIMN.processing.tools.kernels import gaussian_kernel_norm as gaussian_kernel
except ImportError as e:
    print(f"Erro na importação: {e}")
    print("Certifique-se de que a GimnTools está instalada:")
    print("pip install -e .")
    exit(1)
    
def exemplo_manipulacao_imagem():
    """Demonstra criação e manipulação básica de imagens"""
    print("🖼️  Exemplo 1: Manipulação Básica de Imagem")
    print("-" * 50)
    
    # Criar dados de exemplo (imagem sintética)
    tamanho = 128
    x, y = np.meshgrid(np.linspace(-1, 1, tamanho), np.linspace(-1, 1, tamanho))
    data = np.exp(-(x**2 + y**2) / 0.3)  # Gaussiana 2D
    
    # Criar objeto image
    img = image(data)
    img.set_name("exemplo_gaussiana")
    
    print(f"✅ Imagem criada com dimensões: {img.pixels.shape}")
    print(f"✅ Está vazia: {img.is_empty}")
    print(f"✅ Valor mínimo: {img.pixels.min():.3f}")
    print(f"✅ Valor máximo: {img.pixels.max():.3f}")
    
    return img

def exemplo_filtragem():
    """Demonstra aplicação de filtros espaciais"""
    print("\n🔧 Exemplo 2: Filtragem Espacial")
    print("-" * 50)
    
    # Criar imagem com ruído
    tamanho = 64
    x, y = np.meshgrid(np.linspace(-2, 2, tamanho), np.linspace(-2, 2, tamanho))
    signal = np.sin(x) * np.cos(y)
    noise = np.random.normal(0, 0.1, signal.shape)
    imagem_ruidosa = signal + noise
    
    # Criar kernel gaussiano para suavização
    kernel_size = 5
    sigma = 1.0
    kernel = gaussian_kernel(kernel_size,kernel_size, sigma)
    
    print(f"✅ Imagem com ruído criada: {imagem_ruidosa.shape}")
    print(f"✅ Kernel gaussiano: {kernel_size}x{kernel_size}, σ={sigma}")
    
    # Aplicar filtro
    try:
        imagem_filtrada = apply_filter(kernel, imagem_ruidosa)
        print(f"✅ Filtro aplicado com sucesso")
        print(f"✅ Redução de ruído: {np.std(imagem_ruidosa):.3f} → {np.std(imagem_filtrada):.3f}")
        
        return imagem_ruidosa, imagem_filtrada, kernel
    except Exception as e:
        print(f"❌ Erro na filtragem: {e}")
        return None, None, None

def exemplo_visualizacao(imagem_original, imagem_filtrada, kernel):
    """Cria visualizações dos resultados"""
    print("\n📊 Exemplo 3: Visualização")
    print("-" * 50)
    
    if any(x is None for x in [imagem_original, imagem_filtrada, kernel]):
        print("❌ Dados não disponíveis para visualização")
        return
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Exemplo GimnTools - Processamento de Imagem', fontsize=16)
    
    # Imagem original com ruído
    im1 = axes[0, 0].imshow(imagem_original, cmap='viridis')
    axes[0, 0].set_title('Imagem Original (com ruído)')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Imagem filtrada
    im2 = axes[0, 1].imshow(imagem_filtrada, cmap='viridis')
    axes[0, 1].set_title('Imagem Filtrada (suavizada)')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Kernel utilizado
    im3 = axes[1, 0].imshow(kernel, cmap='hot')
    axes[1, 0].set_title('Kernel Gaussiano')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Diferença (ruído removido)
    diferenca = imagem_original - imagem_filtrada
    im4 = axes[1, 1].imshow(diferenca, cmap='RdBu')
    axes[1, 1].set_title('Diferença (ruído removido)')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    
    # Salvar figura
    output_dir = Path("examples/output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "exemplo_basico.png"
    
    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visualização salva em: {output_path}")
    except Exception as e:
        print(f"⚠️  Não foi possível salvar a figura: {e}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"⚠️  Não foi possível exibir a figura: {e}")

def exemplo_estatisticas_imagem(img):
    """Calcula e exibe estatísticas da imagem"""
    print("\n📈 Exemplo 4: Estatísticas da Imagem")
    print("-" * 50)
    
    pixels = img.pixels
    
    # Estatísticas básicas
    media = np.mean(pixels)
    mediana = np.median(pixels)
    desvio = np.std(pixels)
    minimo = np.min(pixels)
    maximo = np.max(pixels)
    
    print(f"📊 Estatísticas da Imagem:")
    print(f"   Média: {media:.4f}")
    print(f"   Mediana: {mediana:.4f}")
    print(f"   Desvio Padrão: {desvio:.4f}")
    print(f"   Mínimo: {minimo:.4f}")
    print(f"   Máximo: {maximo:.4f}")
    print(f"   Forma: {pixels.shape}")
    print(f"   Tipo: {pixels.dtype}")
    
    # Histograma
    try:
        hist, bins = np.histogram(pixels.flatten(), bins=50)
        print(f"   Histograma calculado com {len(bins)-1} bins")
    except Exception as e:
        print(f"⚠️  Erro ao calcular histograma: {e}")

def main():
    """Função principal que executa todos os exemplos"""
    print("🚀 GimnTools - Exemplos Básicos de Uso")
    print("=" * 60)
    
    try:
        # Exemplo 1: Manipulação básica
        img = exemplo_manipulacao_imagem()
        
        # Exemplo 2: Filtragem
        img_ruidosa, img_filtrada, kernel = exemplo_filtragem()
        
        # Exemplo 3: Visualização
        exemplo_visualizacao(img_ruidosa, img_filtrada, kernel)
        
        # Exemplo 4: Estatísticas
        exemplo_estatisticas_imagem(img)
        
        print("\n🎉 Todos os exemplos executados com sucesso!")
        print("📚 Consulte a documentação para mais funcionalidades:")
        print("   https://gimntools.readthedocs.io/")
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        print("🔧 Verifique se todas as dependências estão instaladas")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
